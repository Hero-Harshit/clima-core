"""
Command Line Interface for Clima-Core.
Provides commands for source registry inspection, data auditing, dataset preparation,
model training, evaluation, and artifact inspection.
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
    else:
        print(f"Unknown model: {args.model}")


def cmd_evaluate(args):
    """Display saved evaluation metrics."""
    root = get_project_root()
    eval_file = root / "reports" / "evaluation" / f"{args.model}_evaluation_report.json"
    if not eval_file.exists():
        print(f"Evaluation report not found at {eval_file}. Please run train first.")
        return
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=== Evaluation Summary for {args.model} ({data['version']}) ===")
    print(f"Target: {data['target']} ({data['units']})")
    print(f"Split: {data['split_definition']['test']}")
    print("\nMetrics Comparison on Test Set:")
    for m_name, m_vals in data["metrics"].items():
        if "test" in m_name:
            print(f"  {m_name:25s} | MAE: {m_vals.get('mae')} | RMSE: {m_vals.get('rmse')}")


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
    p_prep.add_argument("--model", required=True, choices=["air_quality", "heat_risk"], help="Target model")

    # train
    p_train = subparsers.add_parser("train", help="Train model and export artifacts")
    p_train.add_argument("--model", required=True, choices=["air_quality", "heat_risk"], help="Model to train")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Display model evaluation report")
    p_eval.add_argument("--model", required=True, choices=["air_quality", "heat_risk"], help="Model to evaluate")

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


if __name__ == "__main__":
    main()
