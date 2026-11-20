# Clima-Core v1.3 — BasinFlow V1 Hydrological Modelling & Routing Report

## Executive Summary

BasinFlow V1 advances Clima-Core from static rainfall-excess estimation into time-dependent catchment hydrodynamics and river reach routing. Built upon the USDA NRCS National Engineering Handbook (NEH-630) Part 630 and Cunge's (1969) hydraulic routing formulation, BasinFlow couples:
1. **SCS Dimensionless Unit Hydrograph Engine:** Translates SCS-CN event runoff excess depth $Q$ (mm) into continuous time-distributed discharge hydrographs $q(t)$ ($\text{m}^3/\text{s}$) with exact mass conservation ($\text{volume error} < 0.001\%$).
2. **Muskingum-Cunge Hydraulic Channel Routing:** Routes catchment flood hydrographs through downstream HydroRIVERS reaches with dynamic kinematic wave celerity ($c_k = \frac{5}{3}v_0$), variable sub-reach discretization ($\Delta x = L / m$) for Courant numerical stability, and peak attenuation tracking.
3. **FloodRisk BasinFlow Challenger Model (30 features):** Incorporates hydrograph peak discharge ($q_p$), time-to-peak ($t_p$), routed river peak ($O_{\text{peak}}$), Strahler stream order, and accumulated upstream catchment runoff into the district inundation classifier.

### Empirical Tournament Result
Evaluated on the exact 36,500 test samples (2022–2023 chronological holdout across 50 Indian cities with 441 verified IIT Delhi IFI disaster events):
- **SCS-CN Champion (23 features):** $\text{ROC-AUC} = 0.8013$ ($95\%\text{ CI } [0.7767, 0.8239]$), $\text{PR-AUC} = 0.1226$ ($95\%\text{ CI } [0.0936, 0.1560]$), $\text{Brier} = 0.1354$.
- **BasinFlow Challenger (30 features):** $\text{ROC-AUC} = 0.7880$ ($95\%\text{ CI } [0.7624, 0.8130]$), $\text{PR-AUC} = 0.1223$ ($95\%\text{ CI } [0.0927, 0.1563]$), $\text{Brier} = 0.1370$.
- **Tournament Verdict:** The BasinFlow Challenger does not exceed the SCS-CN Champion on administrative disaster declaration labels ($\Delta\text{ROC} = -0.0133$, $\Delta\text{PR} = -0.0002$). Therefore, following strict scientific governance, **the 23-feature SCS-CN model is retained as the sole operational Champion**, while BasinFlow V1 is promoted as the platform's **Diagnostic Physical Hydrology Engine**.

---

## 1. SCS Dimensionless Unit Hydrograph Formulation

The unit hydrograph converts instantaneous or uniform rainfall-excess into a time-distributed hydrograph at the catchment outlet.

### Mathematical Formulation
1. **Time of Concentration ($t_c$):** Calculated using the Kirpich / NRCS empirical lag formulation:
   $$t_c = 0.01947 \cdot \frac{L^{0.77}}{S_0^{0.385}} \quad [\text{hours}]$$
   where $L$ is the hydraulic reach length in meters and $S_0$ is the catchment channel bed slope ($\text{m}/\text{m}$).
2. **Time to Peak ($t_p$):**
   $$t_p = \frac{\Delta t}{2} + 0.6 \cdot t_c \quad [\text{hours}]$$
   where $\Delta t$ is the excess rainfall duration ($1.0\text{ hour}$).
3. **Peak Discharge ($q_p$):**
   $$q_p = \frac{0.208 \cdot A \cdot Q}{t_p} \quad [\text{m}^3/\text{s}]$$
   *SI Dimensional Verification:*
   $$\frac{1\text{ mm} \times 1\text{ km}^2}{1\text{ hour}} = \frac{10^{-3}\text{ m} \times 10^6\text{ m}^2}{3600\text{ s}} = \frac{1000}{3600}\text{ m}^3/\text{s} = 0.2778\text{ m}^3/\text{s}$$
   The standard NRCS curvilinear triangular ratio ($\frac{2}{2.67}$) yields:
   $$\frac{2}{2.67} \times 0.2778 = 0.2081 \approx 0.208$$
4. **Dimensionless Coordinates & Mass Conservation:**
   Hydrograph ordinates are sampled from the 33 standard dimensionless ratios $(t / t_p, q / q_p)$ defined in USDA NRCS NEH Part 630 Chapter 16. Total runoff volume is integrated numerically:
   $$V = \int_0^{t_b} q(t) dt \quad [\text{m}^3]$$
   and normalized to match $A \times Q \times 10^3\text{ m}^3$ to within $<0.001\%$ numerical tolerance.

---

## 2. Muskingum-Cunge Hydraulic Channel Routing

BasinFlow implements Cunge's (1969) extension of the Muskingum routing method, where coefficients are derived directly from hydraulic properties rather than empirical calibration.

### Governing Hydraulic Equations
1. **Normal Flow Depth ($y_0$):** Derived from Manning's equation for a wide rectangular channel:
   $$y_0 = \left( \frac{n \cdot Q_0}{B \sqrt{S_0}} \right)^{3/5} \quad [\text{m}]$$
   where $n = 0.035$ (alluvial channel roughness), $B = 25\text{ m}$ (mean channel bottom width), and $Q_0$ is the reference discharge ($0.5 \cdot q_p + Q_{\text{base}}$).
2. **Kinematic Wave Celerity ($c_k$):**
   $$v_0 = \frac{Q_0}{B \cdot y_0}, \quad c_k = \frac{5}{3} v_0 \quad [\text{m}/\text{s}]$$
3. **Storage & Weighting Parameters:**
   $$K = \frac{L}{c_k} \quad [\text{seconds}], \quad X = \frac{1}{2} \left( 1 - \frac{Q_0}{B \cdot S_0 \cdot c_k \cdot L} \right)$$
   $X$ is physically constrained to $[0.05, 0.45]$.
4. **Sub-Reach Discretization for Numerical Stability:**
   To satisfy the Courant stability criterion and prevent spurious negative coefficients or numerical diffusion, reaches are subdivided into $m$ sub-reaches:
   $$m = \left\lceil \frac{K}{\Delta t_{\text{routing}}} \right\rceil, \quad K_{\text{sub}} = \frac{K}{m}$$
5. **Routing Equation:**
   $$O_{t+1} = C_1 I_{t+1} + C_2 I_t + C_3 O_t$$
   where:
   $$C_1 = \frac{\Delta t - 2 K_{\text{sub}} X}{2 K_{\text{sub}} (1 - X) + \Delta t}, \quad C_2 = \frac{\Delta t + 2 K_{\text{sub}} X}{2 K_{\text{sub}} (1 - X) + \Delta t}, \quad C_3 = \frac{2 K_{\text{sub}} (1 - X) - \Delta t}{2 K_{\text{sub}} (1 - X) + \Delta t}$$
   with $C_1 + C_2 + C_3 = 1.0000$.

*Verification Benchmark:* On a $25\text{ km}$ reach carrying a $722.6\text{ m}^3/\text{s}$ inflow hydrograph ($t_p = 1.94\text{ h}$), Muskingum-Cunge yielded $m = 20$ sub-reaches, coefficients $(C_1=0.0483, C_2=0.8794, C_3=0.0723)$, peak outflow of $717.8\text{ m}^3/\text{s}$ ($0.9934$ attenuation), lag time of $1.554\text{ hours}$, and **1.0000 volume conservation ratio**.

---

## 3. Challenger Model Evaluation vs. SCS-CN Champion

| Model Architecture | Features | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Brier Score | Recall (@0.35) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCS-CN LightGBM (Champion)** | 23 | **0.8013** [0.7767, 0.8239] | **0.1226** [0.0936, 0.1560] | **0.1354** | 73.7% | **Champion** |
| **BasinFlow LightGBM (Challenger)**| 30 | 0.7880 [0.7624, 0.8130] | 0.1223 [0.0927, 0.1563] | 0.1370 | 71.4% | Diagnostic Challenger |
| Naive Rain Baseline ($\ge 64.5\text{ mm}$) | 1 | 0.7483 | 0.0521 | 0.2810 | 48.2% | Baseline |

### Feature Importance Analysis in BasinFlow Model
1. `rain_rolling_30d`: $456$ (Long-term soil saturation)
2. `api_14d`: $283$ (Antecedent precipitation index)
3. `rain_rolling_14d`: $264$
4. `drainage_canals_15km`: $232$ (Urban drainage density)
5. `et0_fao_evapotranspiration`: $223$
6. `rain_rolling_7d`: $218$
7. `transit_stations_15km`: $203$
8. `bridges_15km`: $187$
9. `wind_gusts_10m_max`: $185$
10. `population_15km`: $168$
11. `soil_moisture_7_to_28cm_mean`: $157$
12. `hydrograph_peak_discharge_m3s`: $112$
13. `routed_river_peak_m3s`: $98$

*Root-Cause Analysis of Metric Parity:* While unit hydrograph peak flow and routed river discharge are physically faithful representations of channel physics, IIT Delhi IFI flood records represent administrative disaster assistance declarations, which are heavily dominated by 14-to-30 day prolonged antecedent waterlogging and urban pluvial drainage choke rather than purely fluvial peak hydrograph timing. Thus, BasinFlow features add scientific realism without degrading PR-AUC, but do not warrant displacing the champion.

---

## 4. Graph Neural Network (GNN) Streamflow Blocker Declaration

```
STATUS: BLOCKED_AWAITING_CALIBRATED_DISCHARGE_SENSORS
```
- **Context:** Graph Neural Networks (e.g. Spatio-Temporal GNNs or Message Passing Neural Networks on HydroRIVERS topologies) require continuous streamflow discharge time-series ($\text{m}^3/\text{s}$) at gauging stations for supervised loss backpropagation ($L_2$ or NSE loss on $Q_t$).
- **Data Gap:** The open India WRIS / Central Water Commission (CWC) telemetry portals currently publish only instantaneous water level / gauge stage (meters) at telemetry stations. Continuous rating curves ($Q = a(h - h_0)^b$), cross-section geometry, and discharge calibrations are withheld or non-continuous.
- **Scientific Decision:** Training an uncalibrated GNN against synthetic or estimated flows would create spurious precision and violate scientific integrity. BasinFlow retains deterministic physical routing (Muskingum-Cunge) until verified streamflow gauge networks become publicly available.
