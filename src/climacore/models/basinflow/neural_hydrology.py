"""Neural Hydrology Architectures for BasinFlow V2:
1. Standard LSTM: Data-driven recurrent benchmark with static attribute fusion.
2. Mass-Conserving LSTM (MC-LSTM): Physics-constrained recurrent architecture
   guaranteeing mass balance conservation of incoming precipitation.
3. Mass-Conserving LSTM Dual State (MC-LSTM-DS): Two-reservoir architecture
   with dynamic infiltration partition into quickflow and slow baseflow.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.climacore.models.basinflow.metrics import compute_hydrological_metrics

# Strict CPU thread ceiling
torch.set_num_threads(2)


# -----------------------------------------------------------------------------
# 1. Standard LSTM
# -----------------------------------------------------------------------------

class StandardLSTM(nn.Module):
    """Standard LSTM streamflow model with static attribute concatenation."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Ensures non-negative streamflow
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_dim)
        q_pred = self.head(last_hidden).squeeze(-1)  # Shape: (batch_size,)
        return q_pred


# -----------------------------------------------------------------------------
# 2. Mass-Conserving LSTM (MC-LSTM) (Hoedt et al. 2021)
# -----------------------------------------------------------------------------

class MCLSTMCell(nn.Module):
    """Mass-Conserving LSTM Cell.
    Conserves mass by distributing input precipitation via a softmax input gate
    and controlling retention/release via a forget gate.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates take driving meteorological and static features
        self.w_in = nn.Linear(input_dim, hidden_dim)
        self.w_forget = nn.Linear(input_dim, hidden_dim)
        self.w_out = nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        p_t: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step of MC-LSTM:
        x_t: Driving features (batch_size, input_dim)
        p_t: Raw liquid precipitation (batch_size, 1)
        c_prev: Previous storage state (batch_size, hidden_dim)
        """
        # 1. Softmax input gate distributes incoming water mass across stores: sum_k(i_k) = 1
        i_t = F.softmax(self.w_in(x_t), dim=-1)
        added_mass = i_t * p_t  # Preserves sum(added_mass) == p_t

        # 2. Forget gate controls fraction retained in storage
        f_t = torch.sigmoid(self.w_forget(x_t))
        retained_mass = f_t * c_prev
        released_mass = (1.0 - f_t) * c_prev

        # 3. Cell state update (Storage balance: c_t = c_{t-1} + P_t - released)
        c_t = torch.clamp(retained_mass + added_mass, min=0.0, max=10000.0)

        # 4. Outflow routing: fraction of released mass routed to river streamflow
        o_t = torch.sigmoid(self.w_out(x_t))
        q_t = torch.sum(o_t * released_mass, dim=-1)  # (batch_size,)

        return c_t, q_t, released_mass


class MCLSTM(nn.Module):
    """Full sequence Mass-Conserving LSTM."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = MCLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, raw_p: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x shape: (batch_size, seq_len, input_dim)
        # raw_p shape: (batch_size, seq_len)
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize storage to non-negative initial moisture
        c_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        q_series = []
        tot_released = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            p_t = raw_p[:, t:t+1]
            c_t, q_t, rel_t = self.cell(x_t, p_t, c_t)
            q_series.append(q_t)
            tot_released.append(torch.sum(rel_t, dim=-1))

        final_q = q_series[-1]
        diagnostics = {
            "final_storage": c_t,
            "q_series": torch.stack(q_series, dim=1),
        }
        return final_q, diagnostics


# -----------------------------------------------------------------------------
# 3. Mass-Conserving LSTM Dual State (MC-LSTM-DS)
# -----------------------------------------------------------------------------

class MCLSTMDualStateCell(nn.Module):
    """Two-reservoir Mass-Conserving LSTM Cell:
    - Quickflow reservoir (surface/direct storm runoff)
    - Slowflow reservoir (deep groundwater / baseflow)
    With dynamic infiltration partition gate alpha_t.
    """

    def __init__(self, input_dim: int = 10, reservoir_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.dim = reservoir_dim

        # Infiltration gate: partitions P into surface excess (1 - alpha) and infiltration (alpha)
        self.w_infilt = nn.Linear(input_dim, 1)

        # Quickflow gates
        self.w_in_quick = nn.Linear(input_dim, reservoir_dim)
        self.w_forget_quick = nn.Linear(input_dim, reservoir_dim)
        self.w_out_quick = nn.Linear(input_dim, reservoir_dim)

        # Slowflow gates
        self.w_in_slow = nn.Linear(input_dim, reservoir_dim)
        self.w_forget_slow = nn.Linear(input_dim, reservoir_dim)
        self.w_out_slow = nn.Linear(input_dim, reservoir_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        p_t: torch.Tensor,
        c_quick_prev: torch.Tensor,
        c_slow_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Infiltration partitioning: alpha in (0, 1)
        alpha = torch.sigmoid(self.w_infilt(x_t))  # (batch_size, 1)
        p_slow = alpha * p_t                     # Infiltrated precipitation
        p_quick = (1.0 - alpha) * p_t            # Direct overland precipitation

        # 2. Quickflow store update
        i_quick = F.softmax(self.w_in_quick(x_t), dim=-1)
        added_quick = i_quick * p_quick
        f_quick = torch.sigmoid(self.w_forget_quick(x_t))
        rel_quick = (1.0 - f_quick) * c_quick_prev
        c_quick = torch.clamp(f_quick * c_quick_prev + added_quick, min=0.0, max=10000.0)

        # 3. Slowflow store update
        i_slow = F.softmax(self.w_in_slow(x_t), dim=-1)
        added_slow = i_slow * p_slow
        # Slower recession for groundwater (bias towards 1.0)
        f_slow = torch.sigmoid(self.w_forget_slow(x_t) + 1.5)
        rel_slow = (1.0 - f_slow) * c_slow_prev
        c_slow = torch.clamp(f_slow * c_slow_prev + added_slow, min=0.0, max=10000.0)

        # 4. Outflow routing
        o_quick = torch.sigmoid(self.w_out_quick(x_t))
        o_slow = torch.sigmoid(self.w_out_slow(x_t))

        q_quick = torch.sum(o_quick * rel_quick, dim=-1)
        q_slow = torch.sum(o_slow * rel_slow, dim=-1)
        q_total = q_quick + q_slow

        return c_quick, c_slow, q_total, q_quick, q_slow


class MCLSTMDualState(nn.Module):
    """Full sequence Mass-Conserving LSTM with Dual States (MC-LSTM-DS)."""

    def __init__(self, input_dim: int = 10, reservoir_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.dim = reservoir_dim
        self.cell = MCLSTMDualStateCell(input_dim=input_dim, reservoir_dim=reservoir_dim)

    def forward(
        self,
        x: torch.Tensor,
        raw_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        device = x.device

        c_quick = torch.zeros(batch_size, self.dim, device=device)
        c_slow = torch.zeros(batch_size, self.dim, device=device)

        q_tot_series = []
        q_quick_series = []
        q_slow_series = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            p_t = raw_p[:, t:t+1]
            c_quick, c_slow, q_tot, q_q, q_s = self.cell(x_t, p_t, c_quick, c_slow)
            q_tot_series.append(q_tot)
            q_quick_series.append(q_q)
            q_slow_series.append(q_s)

        diagnostics = {
            "final_quick_storage": c_quick,
            "final_slow_storage": c_slow,
            "q_total_series": torch.stack(q_tot_series, dim=1),
            "q_quick_series": torch.stack(q_quick_series, dim=1),
            "q_slow_series": torch.stack(q_slow_series, dim=1),
        }
        return q_tot_series[-1], diagnostics


# -----------------------------------------------------------------------------
# Training and Evaluation Harness for Deep Hydrological Models
# -----------------------------------------------------------------------------

def train_neural_hydrology_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str = "mc_lstm_ds",
    num_epochs: int = 15,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains neural hydrology model with SmoothL1 loss and validation KGE tracking."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.SmoothL1Loss()

    best_val_kge = -999.0
    best_state_dict = None
    history = []

    print(f"Starting CPU training for {model.__class__.__name__} (max 2 threads, epochs={num_epochs})...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            raw_p = batch["raw_p"].to(device)
            target = batch["target_q"].to(device)

            optimizer.zero_grad()
            if isinstance(model, StandardLSTM):
                preds = model(x)
            elif isinstance(model, MCLSTM):
                preds, _ = model(x, raw_p)
            elif isinstance(model, MCLSTMDualState):
                preds, _ = model(x, raw_p)
            else:
                preds = model(x)

            if torch.isnan(target).any() or torch.isnan(x).any():
                continue

            loss = criterion(preds, target)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation evaluation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                raw_p = batch["raw_p"].to(device)
                target = batch["target_q"].numpy()

                if isinstance(model, StandardLSTM):
                    preds = model(x).cpu().numpy()
                elif isinstance(model, MCLSTM):
                    preds, _ = model(x, raw_p)
                    preds = preds.cpu().numpy()
                elif isinstance(model, MCLSTMDualState):
                    preds, _ = model(x, raw_p)
                    preds = preds.cpu().numpy()
                else:
                    preds = model(x).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(target)

        val_metrics = compute_hydrological_metrics(np.array(val_targets), np.array(val_preds))
        val_kge = val_metrics["kge"]
        scheduler.step(val_kge)

        if val_kge > best_val_kge:
            best_val_kge = val_kge
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_kge": val_kge,
            "val_nse": val_metrics["nse"],
            "val_rmse": val_metrics["rmse"],
        })

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val KGE: {val_kge:.3f} | Val NSE: {val_metrics['nse']:.3f}")

    elapsed = time.time() - start_time
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model_name": model.__class__.__name__,
        "best_val_kge": round(best_val_kge, 4),
        "total_training_time_sec": round(elapsed, 2),
        "history": history,
    }


def predict_full_catchment_sequence(
    model: nn.Module,
    df: pd.DataFrame,
    static_vec: np.ndarray,
    seq_len: int = 90,
    target_mask: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Generates continuous streamflow simulation for target indices or full timeline."""
    model.eval()
    n = len(df)
    sim = np.zeros(n, dtype=np.float32)

    # Normalized forcing
    forcing = np.column_stack([
        df["prcp"].values / 50.0,
        df["pet"].values / 10.0,
        (df["tavg"].values - 20.0) / 20.0,
    ]).astype(np.float32)
    static_seq = np.tile(static_vec, (seq_len, 1))

    raw_p = df["prcp"].values.astype(np.float32)

    if target_mask is not None:
        indices = [i for i in range(seq_len, n) if target_mask[i]]
    else:
        indices = list(range(seq_len, n))

    with torch.no_grad():
        batch_size = 128
        for b_start in range(0, len(indices), batch_size):
            b_indices = indices[b_start : b_start + batch_size]
            x_batch = []
            p_batch = []
            for idx in b_indices:
                x_seq = forcing[idx - seq_len : idx]
                x_full = np.hstack([x_seq, static_seq])
                x_batch.append(x_full)
                p_batch.append(raw_p[idx - seq_len : idx])

            x_tensor = torch.from_numpy(np.array(x_batch, dtype=np.float32)).to(device)
            p_tensor = torch.from_numpy(np.array(p_batch, dtype=np.float32)).to(device)

            if isinstance(model, StandardLSTM):
                preds = model(x_tensor).cpu().numpy()
            elif isinstance(model, (MCLSTM, MCLSTMDualState)):
                preds, _ = model(x_tensor, p_tensor)
                preds = preds.cpu().numpy()
            else:
                preds = model(x_tensor).cpu().numpy()

            for idx, p_val in zip(b_indices, preds):
                sim[idx] = max(float(p_val), 0.0)

    return sim
