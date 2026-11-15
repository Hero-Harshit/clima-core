"""
Command Line Interface for Clima-Core.
Provides commands for source registry inspection, data auditing, dataset preparation,
model training, evaluation, and compound risk assessment.
"""

import argparse
import json
import sys
from pathlib import Path

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root, get_data_sources_config
from climacore.data.audit import audit_india_50_cities_dataset
from climacore.models.air_quality.dataset import prepare_air_quality_dataset
from climacore.models.air_quality.train import train_air_quality_model
from climacore.models.heat_risk.dataset import prepare_heat_risk_dataset
from climacore.models.heat_risk.train import train_heat_risk_model
from climacore.models.rain_risk.dataset import prepare_rain_risk_dataset
from climacore.models.rain_risk.train import train_rain_risk_models
from climacore.models.compound_risk.scoring import CompoundRiskEngine
from climacore.models.compound_risk.evaluate import run_compound_risk_evaluation
from climacore.models.climate_master.train import train_climate_master_model
from climacore.models.imd_rain_gauge.train import train_imd_rain_gauge_model
from climacore.models.cpcb_station.train import train_cpcb_station_model
from climacore.models.apparent_heat.train import train_apparent_heat_model
from climacore.models.multi_horizon import train_multi_horizon_models
from climacore.models.uncertainty import train_quantile_uncertainty_models
from climacore.models.rain_risk.extreme_rain import train_extreme_rain_classifier
from climacore.models.heatwave_alert.train import train_heatwave_alert_model
from climacore.models.cpcb_multipollutant.train import train_cpcb_multipollutant_models
from climacore.models.atmospheric_stagnation.train import train_atmospheric_stagnation_model
from climacore.models.uhi_microclimate.train import train_uhi_microclimate_model
from climacore.models.transboundary_smoke.train import train_transboundary_smoke_model
from climacore.models.gale_gust_hazard.train import train_gale_gust_hazard_model
from climacore.models.bioclimatic_clustering.train import train_bioclimatic_clustering_model
from climacore.models.soil_moisture_drought.train import train_soil_moisture_drought_models
from climacore.explainability.engine import generate_explainability_artifacts
from climacore.personas.models import PersonaInput
from climacore.personas.engine import evaluate_persona, evaluate_all_personas

logger = get_logger("climacore.cli")


def cmd_sources(args):
    """List registered data sources from configuration."""
    cfg = get_data_sources_config()
    print("\n=== Registered Clima-Core Data Sources ===")
    for src_id, meta in cfg.get("sources", {}).items():
        print(f"\n[{src_id}] - {meta.get('display_name')}")
        print(f"  Provider: {meta.get('provider')}")
        print(f"  License: {meta.get('license')}")
        print(f"  Value Type: {meta.get('value_type')}")
        print(f"  Spatial: {meta.get('spatial_resolution')}")


def cmd_audit(args):
    """Run data quality and schema audit."""
    root = get_project_root()
    if args.source == "india_weather_air_quality_50_cities":
        report = audit_india_50_cities_dataset(
            raw_dir=root / "data" / "raw" / "india_weather_air_quality",
            output_dir=root / "reports" / "data_audit",
            manifest_dir=root / "data" / "manifests",
        )
        print("\nAudit completed. Summary:")
        print(f"  Validation Passed: {report['validation_passed']}")
        print(f"  Weather Rows: {report['weather_profile']['rows']:,}")
        print(f"  AQ Rows: {report['air_quality_profile']['rows']:,}")
    else:
        print(f"Unknown or unconfigured source: {args.source}")


def cmd_prepare(args):
    """Prepare processed features and dataset for training."""
    if args.model == "air_quality":
        _, feats, meta = prepare_air_quality_dataset()
        print(f"\nAirQuality dataset prepared: {meta['rows']:,} rows, {meta['feature_count']} features.")
    elif args.model == "heat_risk":
        _, feats, meta = prepare_heat_risk_dataset()
        print(f"\nHeatRisk dataset prepared: {meta['rows']:,} rows, {meta['feature_count']} features.")
    elif args.model == "rain_risk":
        _, feats, meta = prepare_rain_risk_dataset()
        print(f"\nRainRisk dataset prepared: {meta['rows']:,} rows, {meta['feature_count']} features.")
    else:
        print(f"Unknown model: {args.model}")


def cmd_train(args):
    """Train configured model and export versioned artifact."""
    if args.model == "air_quality":
        results = train_air_quality_model()
        print("\nAirQuality Model Trained Successfully:")
        print(f"  Test MAE: {results['metrics']['lightgbm_test']['mae']} ug/m3")
        print(f"  Test RMSE: {results['metrics']['lightgbm_test']['rmse']} ug/m3")
        print(f"  Test R2: {results['metrics']['lightgbm_test']['r2_score']}")
    elif args.model == "heat_risk":
        results = train_heat_risk_model()
        print("\nHeatRisk Model Trained Successfully:")
        print(f"  Test MAE: {results['metrics']['lightgbm_test']['mae']} deg C")
        print(f"  Test RMSE: {results['metrics']['lightgbm_test']['rmse']} deg C")
        print(f"  Test R2: {results['metrics']['lightgbm_test']['r2_score']}")
    elif args.model == "rain_risk":
        results = train_rain_risk_models()
        print("\nRainRisk Two-Part Model Trained Successfully:")
        print(f"  Occurrence PR-AUC: {results['classification_metrics']['lightgbm_classifier']['pr_auc']}")
        print(f"  Two-Part Test MAE: {results['regression_metrics']['two_part_hurdle_test']['mae']} mm")
        print(f"  Two-Part Test RMSE: {results['regression_metrics']['two_part_hurdle_test']['rmse']} mm")
        print(f"  Direct Regressor MAE: {results['regression_metrics']['direct_single_regressor_test']['mae']} mm")
    elif args.model == "compound_risk":
        results = run_compound_risk_evaluation()
        print("\nCompoundRisk Engine Evaluation Completed:")
        print(f"  Total 2026 Test Observations: {results['total_test_observations']:,}")
        print(f"  Compounded Co-Occurring Events: {results['compounded_co_occurring_events']['count']:,} ({results['compounded_co_occurring_events']['percentage']}%)")
    elif args.model == "climate_master_44y":
        results = train_climate_master_model()
        print("\n44-Year Climate Master Model Trained Successfully:")
        print(f"  Holdout Test MAE: {results['metrics']['climate_master_lgbm']['mae']} deg C")
        print(f"  Holdout Test RMSE: {results['metrics']['climate_master_lgbm']['rmse']} deg C")
        print(f"  Holdout Test R2: {results['metrics']['climate_master_lgbm']['r2']}")
    elif args.model == "imd_rain_gauge":
        results = train_imd_rain_gauge_model()
        print("\nPhysical IMD Rain Gauge Model Trained Successfully:")
        print(f"  Holdout Test MAE: {results['metrics']['imd_gauge_lgbm']['mae']} mm")
        print(f"  Holdout Test RMSE: {results['metrics']['imd_gauge_lgbm']['rmse']} mm")
        print(f"  Holdout Test R2: {results['metrics']['imd_gauge_lgbm']['r2']}")
    elif args.model == "cpcb_station":
        results = train_cpcb_station_model()
        print("\nPhysical CPCB CAAQMS Station Model Trained Successfully:")
        print(f"  Holdout Test MAE: {results['metrics']['cpcb_station_lgbm']['mae']} ug/m3")
        print(f"  Holdout Test RMSE: {results['metrics']['cpcb_station_lgbm']['rmse']} ug/m3")
        print(f"  Holdout Test R2: {results['metrics']['cpcb_station_lgbm']['r2']}")
    elif args.model == "apparent_heat_stress":
        results = train_apparent_heat_model()
        print("\nApparent Heat Stress Model Trained Successfully:")
        print(f"  Holdout Test MAE: {results['metrics']['apparent_heat_lgbm']['mae']} deg C")
        print(f"  Holdout Test RMSE: {results['metrics']['apparent_heat_lgbm']['rmse']} deg C")
        print(f"  Holdout Test R2: {results['metrics']['apparent_heat_lgbm']['r2']}")
    elif args.model == "multi_horizon":
        results = train_multi_horizon_models()
        print("\nMulti-Horizon (48h & 72h) Models Trained Successfully.")
    elif args.model == "uncertainty":
        results = train_quantile_uncertainty_models()
        print("\nQuantile Uncertainty Models Trained Successfully.")
    elif args.model == "extreme_rain":
        results = train_extreme_rain_classifier()
        print("\nExtreme Heavy Rain Classifier Trained Successfully.")
    elif args.model == "heatwave_alert":
        results = train_heatwave_alert_model()
        print("\nIMD Heatwave Warning Classifier Trained Successfully:")
        print(f"  ROC-AUC: {results['metrics']['heatwave_classifier']['roc_auc']}")
        print(f"  PR-AUC: {results['metrics']['heatwave_classifier']['pr_auc']}")
        print(f"  Recall: {results['metrics']['heatwave_classifier']['recall']*100:.1f}%")
        print(f"  Precision: {results['metrics']['heatwave_classifier']['precision']*100:.1f}%")
    elif args.model == "cpcb_multipollutant":
        results = train_cpcb_multipollutant_models()
        print("\nCPCB Multi-Pollutant (PM10 & NO2) Models Trained Successfully:")
        print(f"  PM10 Test MAE: {results['pm10']['metrics']['lgbm']['mae']} ug/m3")
        print(f"  NO2 Test MAE: {results['no2']['metrics']['lgbm']['mae']} ug/m3")
    elif args.model == "atmospheric_stagnation":
        results = train_atmospheric_stagnation_model()
        print("\nAtmospheric Stagnation & Inversion Model Trained Successfully:")
        print(f"  ROC-AUC: {results['metrics']['stagnation_classifier']['roc_auc']}")
        print(f"  Accuracy: {results['metrics']['stagnation_classifier']['accuracy']*100:.2f}%")
    elif args.model == "uhi_microclimate":
        results = train_uhi_microclimate_model()
        print("\nUrban Heat Island (UHI) Microclimate Model Trained Successfully:")
        print(f"  Holdout Test MAE: {results['metrics']['uhi_lgbm']['mae']} deg C")
        print(f"  Holdout Test RMSE: {results['metrics']['uhi_lgbm']['rmse']} deg C")
        print(f"  Holdout Test R2: {results['metrics']['uhi_lgbm']['r2']}")
    elif args.model == "transboundary_smoke":
        results = train_transboundary_smoke_model()
        print("\nTransboundary Smoke Drift & Wind Advection Model Trained Successfully:")
        print(f"  ROC-AUC: {results['metrics']['advection_classifier']['roc_auc']}")
        print(f"  PR-AUC: {results['metrics']['advection_classifier']['pr_auc']}")
        print(f"  Accuracy: {results['metrics']['advection_classifier']['accuracy']*100:.2f}%")
        print(f"  Recall: {results['metrics']['advection_classifier']['recall']*100:.2f}%")
    elif args.model == "gale_gust_hazard":
        results = train_gale_gust_hazard_model()
        print("\nSevere Gale-Force Wind & Storm Gust Hazard Model Trained Successfully:")
        print(f"  ROC-AUC: {results['metrics']['gale_classifier']['roc_auc']}")
        print(f"  PR-AUC: {results['metrics']['gale_classifier']['pr_auc']}")
        print(f"  Accuracy: {results['metrics']['gale_classifier']['accuracy']*100:.2f}%")
        print(f"  Recall: {results['metrics']['gale_classifier']['recall']*100:.2f}%")
    elif args.model == "bioclimatic_clustering":
        results = train_bioclimatic_clustering_model()
        print("\nSpatial Bioclimatic Climate Clustering Model Trained Successfully:")
        print(f"  Optimal k: {results['optimal_k']}")
        print(f"  Silhouette Score: {results['metrics']['silhouette_score']}")
    elif args.model == "soil_moisture_drought":
        results = train_soil_moisture_drought_models()
        print("\nAgricultural Drought & Root-Zone Soil Moisture Models Trained Successfully:")
        print(f"  Root-Zone Moisture Holdout MAE: {results['metrics']['rootzone_regressor']['mae']} m3/m3 (R2 = {results['metrics']['rootzone_regressor']['r2']})")
        print(f"  Flash Drought Hazard Holdout ROC-AUC: {results['metrics']['flash_drought_classifier']['roc_auc']}, Accuracy: {results['metrics']['flash_drought_classifier']['accuracy']*100:.2f}%")
    elif args.model == "shap_explainability":
        results = generate_explainability_artifacts()
        print("\nSHAP Explainability Artifacts Generated Successfully.")
    else:
        print(f"Unknown model: {args.model}")


def cmd_evaluate(args):
    """Display saved evaluation metrics."""
    root = get_project_root()
    report_mapping = {
        "climate_master_44y": "climate_master_evaluation_report.json",
        "apparent_heat_stress": "apparent_heat_evaluation_report.json",
    }
    filename = report_mapping.get(args.model, f"{args.model}_evaluation_report.json")
    eval_file = root / "reports" / "evaluation" / filename
    if not eval_file.exists():
        print(f"Evaluation report not found at {eval_file}. Please run train first.")
        return
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.model == "compound_risk":
        print(f"\n=== Evaluation Summary for CompoundRisk ({data.get('version', 'v0.1.0')}) ===")
        print(f"Total Observations: {data['total_test_observations']:,}")
        print("\nSeverity Distribution:")
        for cat, cnt in data["severity_distribution"].items():
            print(f"  {cat:15s}: {cnt:,} days")
        print(f"\nMulti-Hazard Co-Occurring Compounded Events: {data['compounded_co_occurring_events']['count']:,} ({data['compounded_co_occurring_events']['percentage']}%)")
        print("\nDominant Drivers:")
        for drv, cnt in data["dominant_hazard_drivers"].items():
            print(f"  {drv:25s}: {cnt:,} days")
        print("\nTop 5 Highest Average Risk Cities (2026):")
        for city, score in data["highest_risk_cities_avg_2026"].items():
            print(f"  {city:20s}: {score}/100")
        return

    if args.model == "soil_moisture_drought":
        print(f"\n=== Evaluation Summary for {data.get('model_name', args.model)} ===")
        print(f"Date Span: {data.get('date_span')}")
        print(f"Cities Covered: {data.get('cities_count')}")
        print(f"Training Rows: {data.get('training_data_rows', 0):,}")
        print(f"Holdout Test Period: {data.get('test_holdout_period')} ({data.get('test_rows', 0):,} rows)")
        print("\nRoot-Zone Regressor Metrics:")
        for k, v in data["metrics"]["rootzone_regressor"].items():
            print(f"  {k:25s}: {v}")
        print("\nFlash Drought Classifier Metrics:")
        for k, v in data["metrics"]["flash_drought_classifier"].items():
            print(f"  {k:25s}: {v}")
        return

    if args.model in ["climate_master_44y", "imd_rain_gauge", "cpcb_station", "apparent_heat_stress", "heatwave_alert", "atmospheric_stagnation", "uhi_microclimate", "transboundary_smoke", "gale_gust_hazard"]:
        print(f"\n=== Evaluation Summary for {data.get('model_name', args.model)} ===")
        print(f"Date Span: {data.get('date_span')}")
        print(f"Cities Covered: {data.get('cities_count')}")
        print(f"Training Rows: {data.get('training_data_rows', 0):,}")
        print(f"Holdout Test Period: {data.get('test_holdout_period')} ({data.get('test_rows', 0):,} rows)")
        print("\nModel Metrics Comparison on Holdout Test:")
        for m_name, m_vals in data.get("metrics", {}).items():
            print(f"  {m_name:30s} | {m_vals}")
        print(f"\nArtifact SHA256: {data.get('artifact_sha256')}")
        return

    if args.model == "bioclimatic_clustering":
        print(f"\n=== Evaluation Summary for {data.get('model_name')} ===")
        print(f"Normals Span: {data.get('climate_normals_span')} across {data.get('cities_count')} cities")
        print(f"Optimal Clusters (k): {data.get('optimal_k')}")
        print(f"Silhouette Score: {data['metrics'].get('silhouette_score')}")
        print(f"Davies-Bouldin Score: {data['metrics'].get('davies_bouldin_score')}")
        print(f"Reload Parity: KMeans diff = {data['metrics'].get('kmeans_reload_parity_diff')}, GMM diff = {data['metrics'].get('gmm_reload_parity_diff')}")
        print("\nRegime Assignments:")
        for cid, reg in data.get("regime_assignments", {}).items():
            print(f"  Cluster {cid}: {reg}")
        return

    if args.model == "cpcb_multipollutant":
        print(f"\n=== Evaluation Summary for {data.get('model_name')} ===")
        print(f"Date Span: {data.get('date_span')} across {data.get('cities_count')} cities")
        print(f"Holdout Test: {data.get('test_holdout_period')}")
        print(f"\nPM10 Model: Test MAE = {data['pm10']['metrics']['lgbm']['mae']} ug/m3, RMSE = {data['pm10']['metrics']['lgbm']['rmse']} ug/m3 (Persistence MAE: {data['pm10']['metrics']['persistence']['mae']} ug/m3)")
        print(f"NO2 Model: Test MAE = {data['no2']['metrics']['lgbm']['mae']} ug/m3, RMSE = {data['no2']['metrics']['lgbm']['rmse']} ug/m3 (Persistence MAE: {data['no2']['metrics']['persistence']['mae']} ug/m3)")
        return

    print(f"\n=== Evaluation Summary for {args.model} ({data.get('version', 'v0.1.0')}) ===")
    print(f"Target: {data.get('target', 'N/A')} ({data.get('units', '')})")
    print(f"Split: {data.get('split_definition', {}).get('test', 'Chronological Holdout')}")
    if "regression_metrics" in data:
        print("\nRegression Metrics Comparison on Test Set:")
        for m_name, m_vals in data["regression_metrics"].items():
            print(f"  {m_name:30s} | MAE: {m_vals.get('mae')} | RMSE: {m_vals.get('rmse')} | R2: {m_vals.get('r2_score')}")
    elif "metrics" in data:
        print("\nMetrics Comparison on Test Set:")
        for m_name, m_vals in data["metrics"].items():
            if "test" in m_name:
                print(f"  {m_name:25s} | MAE: {m_vals.get('mae')} | RMSE: {m_vals.get('rmse')}")


def cmd_assess(args):
    """Run an on-demand compound risk calculation with custom values."""
    engine = CompoundRiskEngine()
    res = engine.calculate_single(
        pm2_5_pred=args.pm25,
        temp_max_pred=args.temp,
        precip_pred=args.rain,
        city=args.city,
        date=args.date,
    )
    print(f"\n=== Clima-Core Compound Risk Assessment ===")
    print(f"City: {res['city']} | Date: {res['date']}")
    print(f"Compound Risk Score: {res['compound_risk_score']} / 100 [{res['severity_category']}]")
    print(f"Dominant Driver: {res['dominant_driver']}")
    if res['is_compounded_event']:
        print(f"Multi-Hazard Compounding Active: Multiplier = {res['compound_multiplier']}x")
    print("\nSub-Hazard Breakdown:")
    for h_name, h_info in res["sub_hazards"].items():
        print(f"  {h_name:15s} | Raw: {h_info['raw_value']} {h_info['unit']} | Score: {h_info['score_0_100']}/100 [{h_info['category']}] | Contribution: {h_info['contribution_pct']}%")
    if res["advisories"]:
        print("\nActive Advisories:")
        for adv in res["advisories"]:
            print(f"  - {adv}")

    # Optional Persona Tailored Intelligence
    if getattr(args, "persona", None):
        p_input = PersonaInput(
            city=args.city,
            date=args.date,
            temp_max=args.temp,
            precipitation_mm=args.rain,
            pm2_5=args.pm25,
            rain_probability=0.8 if args.rain >= 1.0 else 0.1,
            imd_heatwave_alert=2 if args.temp >= 44.0 else (1 if args.temp >= 40.0 else 0),
        )
        if args.persona == "all":
            all_res = evaluate_all_personas(p_input)
            print("\n=== Tailored Multi-Persona Impact Profiles ===")
            for pid, pdata in all_res.items():
                print(f"\n[{pdata['persona_title']}] - STATUS: {pdata['status_level']} (Impact: {pdata['impact_score']}/100)")
                for k, v in pdata["key_metrics"].items():
                    print(f"    • {k}: {v}")
                if pdata["operational_restrictions"] and pdata["operational_restrictions"] != ["No administrative restrictions in effect."]:
                    print("    ⚠️ RESTRICTIONS:")
                    for r in pdata["operational_restrictions"]:
                        print(f"      - {r}")
        else:
            p_res = evaluate_persona(args.persona, p_input)
            print(f"\n=== Tailored Persona Intelligence: {p_res.persona_title} ===")
            print(f"Status: {p_res.status_level} | Impact Score: {p_res.impact_score} / 100")
            print("\nKey Role Metrics:")
            for k, v in p_res.key_metrics.items():
                print(f"  • {k}: {v}")
            if p_res.operational_restrictions:
                print("\nOperational Restrictions:")
                for r in p_res.operational_restrictions:
                    print(f"  - {r}")
            if p_res.advisories:
                print("\nRole-Specific Advisories:")
                for a in p_res.advisories:
                    print(f"  - {a}")


def main():
    parser = argparse.ArgumentParser(prog="climacore", description="Clima-Core ML CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sources
    subparsers.add_parser("sources", help="List registered data sources")

    # audit
    p_audit = subparsers.add_parser("audit", help="Run audit on data source")
    p_audit.add_argument("--source", default="india_weather_air_quality_50_cities", help="Source ID to audit")

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Prepare features and clean dataset")
    p_prep.add_argument("--model", required=True, choices=["air_quality", "heat_risk", "rain_risk"], help="Target model")

    # train
    p_train = subparsers.add_parser("train", help="Train model and export artifacts")
    p_train.add_argument(
        "--model",
        required=True,
        choices=[
            "air_quality", "heat_risk", "rain_risk", "compound_risk",
            "climate_master_44y", "imd_rain_gauge", "cpcb_station", "apparent_heat_stress",
            "multi_horizon", "uncertainty", "extreme_rain",
            "heatwave_alert", "cpcb_multipollutant", "atmospheric_stagnation", "shap_explainability",
            "uhi_microclimate", "transboundary_smoke", "gale_gust_hazard", "bioclimatic_clustering",
            "soil_moisture_drought"
        ],
        help="Model to train/evaluate"
    )

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Display model evaluation report")
    p_eval.add_argument(
        "--model",
        required=True,
        choices=[
            "air_quality", "heat_risk", "rain_risk", "compound_risk",
            "climate_master_44y", "imd_rain_gauge", "cpcb_station", "apparent_heat_stress",
            "heatwave_alert", "cpcb_multipollutant", "atmospheric_stagnation",
            "uhi_microclimate", "transboundary_smoke", "gale_gust_hazard", "bioclimatic_clustering",
            "soil_moisture_drought"
        ],
        help="Model to evaluate"
    )

    # assess (on-demand compound scoring)
    p_assess = subparsers.add_parser("assess", help="Assess compound risk for given values")
    p_assess.add_argument("--city", default="Delhi", help="City name")
    p_assess.add_argument("--date", default="2026-05-27", help="Date YYYY-MM-DD")
    p_assess.add_argument("--pm25", type=float, required=True, help="PM2.5 concentration in ug/m3")
    p_assess.add_argument("--temp", type=float, required=True, help="Max temperature in deg C")
    p_assess.add_argument("--rain", type=float, default=0.0, help="Precipitation in mm")
    p_assess.add_argument(
        "--persona",
        default=None,
        choices=[
            "student", "farmer", "gig_worker", "commuter", "healthcare", "city_official",
            "maritime", "industry", "aviation", "water", "insurance", "cold_chain", "tourism",
            "all"
        ],
        help="Generate tailored persona intelligence across 13 economic sectors"
    )

    args = parser.parse_args()
    if args.command == "sources":
        cmd_sources(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "assess":
        cmd_assess(args)


if __name__ == "__main__":
    main()
