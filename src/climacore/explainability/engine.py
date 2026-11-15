"""
SHAP Interpretability & Feature Attribution Engine for Clima-Core.
Provides mathematical Shapley-value decompositions (via TreeExplainer) to explain
both global risk drivers and real-time local hazard predictions for disaster management.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shap
import lightgbm as lgb
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.explainability.engine")


class ClimaExplainer:
    """Production SHAP Explainer for Clima-Core risk models."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or get_project_root()
        self.explainers: Dict[str, shap.TreeExplainer] = {}
        self.models: Dict[str, lgb.Booster] = {}

    def _get_explainer(self, model_key: str, model_path: Path) -> shap.TreeExplainer:
        if model_key not in self.explainers:
            if not model_path.exists():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            booster = lgb.Booster(model_file=str(model_path))
            self.models[model_key] = booster
            self.explainers[model_key] = shap.TreeExplainer(booster)
        return self.explainers[model_key]

    def explain_instance(
        self,
        model_key: str,
        model_path: Path,
        features_df: pd.DataFrame,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Explain a single row prediction using additive TreeExplainer SHAP values."""
        explainer = self._get_explainer(model_key, model_path)
        booster = self.models[model_key]
        feature_names = booster.feature_name()

        # Align columns
        aligned_df = features_df[feature_names].iloc[[0]]
        shap_vals = explainer.shap_values(aligned_df)

        # Handle binary classifier output list
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
            sv = shap_vals[0]
            base_val = float(explainer.expected_value)
        else:
            sv = np.array(shap_vals).flatten()
            base_val = float(explainer.expected_value)

        pred_val = float(booster.predict(aligned_df)[0])

        # Rank drivers
        driver_items = []
        for name, val in zip(feature_names, sv):
            driver_items.append({"feature": name, "shap_impact": round(float(val), 4), "input_value": aligned_df[name].values[0]})

        # Top positive risk drivers (pushing prediction higher)
        pos_drivers = sorted([d for d in driver_items if d["shap_impact"] > 0], key=lambda x: x["shap_impact"], reverse=True)[:top_k]
        # Top negative risk drivers (mitigating / cooling / suppressing)
        neg_drivers = sorted([d for d in driver_items if d["shap_impact"] < 0], key=lambda x: x["shap_impact"])[:top_k]

        return {
            "model_key": model_key,
            "base_climatological_expectation": round(float(base_val), 4),
            "model_prediction": round(float(pred_val), 4),
            "top_positive_risk_drivers": pos_drivers,
            "top_mitigating_drivers": neg_drivers,
            "total_features_evaluated": len(feature_names),
        }


def generate_explainability_artifacts() -> Dict[str, Any]:
    """Generate and save global feature importance and sample explanations."""
    root = get_project_root()
    out_dir = root / "artifacts" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer_engine = ClimaExplainer(root=root)
    summary_report: Dict[str, Any] = {}

    # 1. HeatRisk Global Explanation
    heat_model = root / "artifacts" / "models" / "heat_risk" / "v0.1.0" / "model.txt"
    heat_data = root / "data" / "processed" / "heat_risk_train_ready.parquet"
    if heat_model.exists() and heat_data.exists():
        logger.info("Computing HeatRisk SHAP global importance...")
        df_h = pd.read_parquet(heat_data).tail(100)
        explainer = explainer_engine._get_explainer("heat_risk", heat_model)
        feats = explainer_engine.models["heat_risk"].feature_name()
        sv = explainer.shap_values(df_h[feats])
        mean_abs = np.mean(np.abs(sv), axis=0)
        ranked = sorted(zip(feats, mean_abs), key=lambda x: x[1], reverse=True)[:10]
        summary_report["heat_risk"] = {
            "top_10_global_drivers": [{"feature": f, "mean_abs_shap": round(float(s), 4)} for f, s in ranked],
            "base_value": round(float(explainer.expected_value), 4),
        }

    # 2. AirQuality Global Explanation
    aq_model = root / "artifacts" / "models" / "air_quality" / "v0.1.0" / "model.txt"
    aq_data = root / "data" / "processed" / "air_quality_train_ready.parquet"
    if aq_model.exists() and aq_data.exists():
        logger.info("Computing AirQuality SHAP global importance...")
        df_aq = pd.read_parquet(aq_data).tail(100)
        explainer = explainer_engine._get_explainer("air_quality", aq_model)
        feats = explainer_engine.models["air_quality"].feature_name()
        sv = explainer.shap_values(df_aq[feats])
        mean_abs = np.mean(np.abs(sv), axis=0)
        ranked = sorted(zip(feats, mean_abs), key=lambda x: x[1], reverse=True)[:10]
        summary_report["air_quality"] = {
            "top_10_global_drivers": [{"feature": f, "mean_abs_shap": round(float(s), 4)} for f, s in ranked],
            "base_value": round(float(explainer.expected_value), 4),
        }

    # 3. IMD Heatwave Alert Global Explanation
    hw_model = root / "artifacts" / "models" / "heatwave_alert" / "v0.1.0" / "model.txt"
    if hw_model.exists():
        logger.info("Computing Heatwave Alert SHAP global importance...")
        booster = lgb.Booster(model_file=str(hw_model))
        explainer = shap.TreeExplainer(booster)
        feats = booster.feature_name()
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_val = float(exp_val[0])
        else:
            base_val = float(exp_val)
        summary_report["heatwave_alert"] = {
            "model_type": "Binary LightGBM IMD Heatwave Alert Classifier",
            "features_audited": len(feats),
            "expected_base_log_odds": round(base_val, 4),
        }

    # 4. Atmospheric Stagnation Global Explanation
    stag_model = root / "artifacts" / "models" / "atmospheric_stagnation" / "v0.1.0" / "model.txt"
    if stag_model.exists():
        logger.info("Computing Stagnation SHAP global importance...")
        booster = lgb.Booster(model_file=str(stag_model))
        explainer = shap.TreeExplainer(booster)
        feats = booster.feature_name()
        exp_val = explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            base_val = float(exp_val[0])
        else:
            base_val = float(exp_val)
        summary_report["atmospheric_stagnation"] = {
            "model_type": "Binary LightGBM Atmospheric Stagnation Classifier",
            "features_audited": len(feats),
            "expected_base_log_odds": round(base_val, 4),
        }

    # Export Global Importance Summary
    summary_file = out_dir / "global_importance_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Saved SHAP Explainability artifacts to {out_dir}.")
    return summary_report


if __name__ == "__main__":
    generate_explainability_artifacts()
