# Clima-Core v1.3 — CompoundRisk V2 Cascading Multi-Hazard Report

## Executive Summary

CompoundRisk V2 establishes a multi-sectoral cascading risk intelligence framework for Clima-Core. While bivariate Archimedean copula modeling (Gumbel, Clayton, Frank) provides tail-dependence estimation for pairs of concurrent extremes (such as heat and power demand, or rainfall and storm surge), copulas alone cannot capture **sequential, multi-tier infrastructure and socio-ecological cascading failures**.

CompoundRisk V2 synthesizes:
1. **Preserved Archimedean Copula Tail Modeling:** Retains full bivariate extreme-value copula fitting, Cramér-von Mises distance family selection, parametric bootstrap uncertainty quantification, and joint "AND" / "OR" return period estimations.
2. **Directed Multi-Pathway Cascading Failure Networks:** Formulates explicit probabilistic directed graphs across three canonical disaster propagation chains:
   - **Pathway A (Thermal-Grid-Health Cascade):** Extreme heatwave $\rightarrow$ transformer derating $\rightarrow$ substation trip $\rightarrow$ municipal pumping failure $\rightarrow$ informal settlement casualty surge.
   - **Pathway B (Drought-Smoke-Respiratory Cascade):** Agricultural soil moisture drought $\rightarrow$ biomass fire advection $\rightarrow$ boundary layer inversion trap $\rightarrow$ toxic PM2.5 spike $\rightarrow$ emergency healthcare surge.
   - **Pathway C (Deluge-Hydrodynamics-Lifeline Cascade):** Antecedent catchment saturation $\rightarrow$ rainfall burst $\rightarrow$ unit hydrograph runoff peak $\rightarrow$ fluvial channel surcharge $\rightarrow$ arterial transport and bridge severance.
3. **Composite Cascading Compound Score ($C_{\text{cascade}} \in [0, 100]$):** Integrates copula joint exceedance probability ($P_{\text{AND}}$) and downstream pathway propagation probability ($P_{\text{cascade}}$).
4. **Bayesian Uncertainty Corridors:** Derives explicit $95\%$ credible intervals $[C_{\text{lower}}, C_{\text{upper}}]$ via Beta conjugate posterior formulation.

---

## 1. Limitations of Pure Copula Dependency in Complex Disasters

Bivariate Archimedean copulas evaluate the joint probability of two simultaneous variables:
$$P(U > u, V > v) = 1 - u - v + C(u, v)$$
However, real-world compounding disasters rarely occur as isolated concurrent pairs. Instead, an initial meteorological trigger propagates through interconnected physical, electrical, and human systems over distinct time lags:
- A heatwave triggers power grid stress not only through increased consumer air conditioning demand, but through physical transformer cooling capacity degradation.
- A substation trip subsequently shuts down municipal water treatment and booster pumps, leaving high-density informal settlements without potable water during $44^\circ\text{C}$ daytime peaks, accelerating heat stroke fatalities.
- Pure copula modeling omits these multi-stage conditional dependencies.

---

## 2. Multi-Pathway Directed Cascading Graph Architecture

### 2.1 Pathway A: Thermal-Grid-Health Cascade
```
[ Ambient Tmax & Tropical Night ]
               │
               ▼
 [ Transformer Thermal Derating ] ───► [ Grid Deficit Probability ]
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                        [ Substation Trip ]
                                  │
                                  ▼
                  [ Municipal Pumping Disruption ]
                                  │
                                  ▼
                [ Informal Cluster Casualty Surge ]
```
- **Node A1 (Thermal Overheat):** $P(\text{Heat}) = \sigma(0.40(T_{\max} - 40.0) + 1.2 \cdot \mathbb{I}_{\text{trop\_night}})$
- **Node A2 (Transformer Derating):** $P(\text{Derate}) = \text{clip}(0.40 P(\text{Heat}) + 0.35 \mathbb{I}_{\text{trop\_night}} + 0.25 P_{\text{grid\_deficit}}, 0, 0.98)$
- **Node A3 (Substation Trip):** $P(\text{Trip}) = 1 - (1 - P(\text{Derate}))(1 - P_{\text{grid\_deficit}})$
- **Node A4 (Pumping Outage):** $P(\text{Pump}) = 0.85 P(\text{Trip})$
- **Node A5 (Casualty Surge):** $P(\text{Casualty}) = \text{clip}(0.35 P(\text{Pump}) + 0.45 \frac{\text{Pop}_{\text{dense}}}{10^6} P(\text{Heat}) + 0.20 \mathbb{I}_{\text{trop\_night}}, 0, 0.95)$
- **End-to-End Propagation:** $P_{\text{Pathway A}} = \left( P(\text{Heat}) \cdot P(\text{Trip}) \cdot P(\text{Casualty}) \right)^{1/3}$

### 2.2 Pathway B: Drought-Smoke-Respiratory Cascade
```
[ Soil Moisture Deficit (DroughtCascade V2) ]
                      │
                      ▼
        [ Biomass Fire Ignition & FRP ] ───► [ Wind Advection Alignment ]
                      │                                    │
                      └────────────────┬───────────────────┘
                                       ▼
                       [ Downwind Smoke Plume Influx ]
                                       │
                                       ▼
                      [ Nocturnal Inversion Stagnation ]
                                       │
                                       ▼
                     [ Toxic PM2.5 Concentration Spike ]
                                       │
                                       ▼
                     [ Emergency Respiratory Burden ]
```
- **Node B1 (Agricultural Drought):** Sourced directly from `DroughtCascadeV2` state probability $P(Z_t = 2)$.
- **Node B2 (Fire Advection Influx):** $P(\text{Fire}) = \text{clip}(0.60 P(Z_t=2) + 0.40 \min(\frac{T_{\text{smoke}}}{100}, 1), 0, 0.95)$
- **Node B3 (Inversion Trap):** $P(\text{Inversion}) = \sigma(0.08 \cdot \text{StagnationIndex} - 1.5)$
- **Node B4 (Severe PM2.5):** $P(\text{PM2.5}) = \min(\frac{\text{PM2.5}}{250.0}, 1.0)$
- **Node B5 (Respiratory Crisis):** $P(\text{Resp}) = \text{clip}(0.50 P(\text{Inversion}) P(\text{PM2.5}) + 0.35 P(\text{Fire}) + 0.15 \frac{\text{Pop}}{2.5\times 10^6}, 0, 0.96)$
- **End-to-End Propagation:** $P_{\text{Pathway B}} = \left( P(\text{Fire}) \cdot P(\text{Inversion}) \cdot P(\text{Resp}) \right)^{1/3}$

### 2.3 Pathway C: Deluge-Hydrodynamics-Lifeline Cascade
```
[ Antecedent Soil Saturation ] ───► [ Extreme Cloudburst / Rainfall ]
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                   [ Unit Hydrograph Runoff Peak (qp) ]
                                  │
                                  ▼
               [ Muskingum-Cunge Fluvial Reach Choke (Opeak) ]
                                  │
                                  ▼
                 [ Arterial Roadway Submergence ]
                                  │
                                  ▼
               [ Critical Emergency Lifeline Severance ]
```
- **Node C1 (Deluge Trigger):** $P(\text{Deluge}) = \sigma(0.08(P_{\text{rain}} - 50.0))$
- **Node C2 (Runoff Surge):** $P(\text{Runoff}) = \text{clip}(0.70 P_{\text{flood\_hazard}} + 0.30 \min(\frac{q_p}{500}, 1), 0, 0.98)$
- **Node C3 (Fluvial Surcharge):** $P(\text{Choke}) = \text{clip}(0.50 P(\text{Runoff}) + 0.50 \min(\frac{O_{\text{peak}}}{400}, 1), 0, 0.95)$
- **Node C4 (Road Submergence):** $P(\text{Road}) = \text{clip}(P(\text{Choke}) \cdot (1.2 - \min(\frac{\text{Canals}}{\text{Roads}}, 0.8)), 0, 0.96)$
- **Node C5 (Lifeline Severance):** $P(\text{Severance}) = \text{clip}(0.60 P(\text{Road}) + 0.25 \frac{\text{Bridges}}{50} P(\text{Deluge}) + 0.15, 0, 0.95)$
- **End-to-End Propagation:** $P_{\text{Pathway C}} = \left( P(\text{Deluge}) \cdot P(\text{Choke}) \cdot P(\text{Severance}) \right)^{1/3}$

---

## 3. Composite Cascading Compound Score & Uncertainty Corridors

The composite cascading index $C_{\text{cascade}} \in [0, 100]$ fuses the statistical copula tail probability with the physical cascading propagation likelihood:
$$C_{\text{cascade}} = 100 \times \left[ 0.35 \cdot \min(1.0, 10 \cdot P_{\text{AND}}) + 0.65 \cdot \max_{p \in \{\text{A, B, C}\}} P_{\text{pathway}}(p) \right]$$

### Bayesian Credible Interval Corridors
To ensure risk analysts do not rely on point estimates, CompoundRisk V2 computes a $95\%$ credible interval $[C_{\text{lower}}, C_{\text{upper}}]$ using a Beta distribution parameterized by the composite probability:
$$\alpha_{\text{post}} = 2.0 + 30 \cdot P_{\text{composite}}, \quad \beta_{\text{post}} = 2.0 + 30 \cdot (1 - P_{\text{composite}})$$
$$C_{\text{lower}} = 100 \cdot F_{\text{Beta}}^{-1}(0.025; \alpha_{\text{post}}, \beta_{\text{post}}), \quad C_{\text{upper}} = 100 \cdot F_{\text{Beta}}^{-1}(0.975; \alpha_{\text{post}}, \beta_{\text{post}})$$

---

## 4. Empirical Case Demonstration: Severe Summer Heat & Atmospheric Smog in Delhi

A simulation of an extreme compound crisis ($T_{\max} = 44.5^\circ\text{C}$, Tropical Night $T_{\min} = 31.0^\circ\text{C}$, $\text{PM2.5} = 310\text{ }\mu\text{g/m}^3$, grid deficit probability $= 0.45$, drought probability $= 0.75$, copula $P_{\text{AND}} = 0.045$):
- **Compound Cascade Score:** $\mathbf{71.0}$ (Corridor: $[52.2, 82.7]$)
- **Dominant Pathway:** Pathway B (Drought $\rightarrow$ Fire $\rightarrow$ Inversion $\rightarrow$ Smog Crisis, $P = 0.8498$)
- **Secondary Pathway:** Pathway A (Thermal $\rightarrow$ Grid Deficit $\rightarrow$ Pumping Collapse, $P = 0.8197$)
- **Tertiary Pathway:** Pathway C (Deluge/Fluvial, $P = 0.0370$)
- **Governing Policy Action:** `ACTIVATE_FULL_MULTI_AGENCY_CASCADE_PROTOCOL`

---

## 5. Artifact Registry & Operational Deployment

- **Module Path:** `src/climacore/analytics/compound_risk_v2.py`
- **Exposed APIs:** `CompoundRiskV2`, `CompoundCopulaEngine`, `CopulaFamily`, `JointReturnPeriodResult`
- **Operational Integration:** Deployed into the Clima-Core persona decision engine (DISCOM grid operators, Municipal Disaster Commissioners, Chief Medical Officers).
