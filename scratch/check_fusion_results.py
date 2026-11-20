import json

with open('reports/v1_3/forecast_fusion_evaluation.json') as f:
    d = json.load(f)

print('=== TMAX COMPARISON ===')
for k, v in d['tmax'].items():
    cov = v.get('coverage_80', '-')
    print(f"{k:28s}: MAE={v.get('mae'):.4f}, RMSE={v.get('rmse'):.4f}, Bias={v.get('bias'):.4f}, 80%Cov={cov}")

print('\n=== TMIN COMPARISON ===')
for k, v in d['tmin'].items():
    cov = v.get('coverage_80', '-')
    print(f"{k:28s}: MAE={v.get('mae'):.4f}, RMSE={v.get('rmse'):.4f}, Bias={v.get('bias'):.4f}, 80%Cov={cov}")

print('\n=== PRECIP COMPARISON ===')
for k, v in d['precip'].items():
    brier = v.get('brier', '-')
    print(f"{k:28s}: MAE={v.get('mae'):.4f}, RMSE={v.get('rmse'):.4f}, Bias={v.get('bias'):.4f}, Brier={brier}")
