from climacore.analytics.compound_risk_v2 import CompoundRiskV2

cr = CompoundRiskV2()
res = cr.evaluate_cascading_network(
    city='Delhi',
    temp_max_c=44.5,
    temp_min_c=31.0,
    rainfall_mm=0.0,
    pm2_5_ug_m3=310.0,
    is_tropical_night=True,
    grid_deficit_prob=0.45,
    flood_hazard_prob=0.05,
    drought_regime_prob=0.75,
    smoke_advection_index=85.0,
    stagnation_index=45.0,
    hydrograph_peak_m3s=0.0,
    routed_river_peak_m3s=0.0,
    copula_p_and=0.045
)
print('City:', res['city'])
print('Compound Cascade Score:', res['compound_cascade_score'], 'CI:', res['uncertainty_corridor_95ci'])
print('Dominant Pathway:', res['dominant_pathway'])
print('Policy Action:', res['governing_policy_action'])
for k, v in res['pathways'].items():
    print(f"  {k:35s}: Prop Prob={v['end_to_end_propagation_prob']} ({v['severity_tier']})")
