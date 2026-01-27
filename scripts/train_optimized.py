#!/usr/bin/env python3
"""
Optimized FastText training script for Croatian language models.
Trains multiple model configurations and saves results for evaluation.

Usage:
    python train_optimized.py --tier 1        # Train Tier 1 models only
    python train_optimized.py --tier 2        # Train Tier 2 models only
    python train_optimized.py --all           # Train all models
    python train_optimized.py --config 0      # Train specific config by index
"""

import argparse
import fasttext
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CORPUS_PATH = PROJECT_DIR / "data" / "corpus_clean.txt"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Tier 0: Quick configurations (~2-3h each instead of ~11h)
TIER0_PARAMS = [
    {
        "name": "sg_d300_ws10_e5_mc2",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 5,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d200_ws10_e5_mc2",
        "model": "skipgram",
        "dim": 200,
        "ws": 10,
        "epoch": 5,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d300_ws5_e5_mc2",
        "model": "skipgram",
        "dim": 300,
        "ws": 5,
        "epoch": 5,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "cbow_d300_ws10_e5_mc2",
        "model": "cbow",
        "dim": 300,
        "ws": 10,
        "epoch": 5,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
]

# Tier 1: High priority configurations (based on Seminar 2 findings)
TIER1_PARAMS = [
    {
        "name": "sg_d300_ws10_e10_mc2",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d300_ws10_e15_mc2",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 15,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d300_ws15_e10_mc2",
        "model": "skipgram",
        "dim": 300,
        "ws": 15,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d300_ws10_e10_mc2_minn2_maxn7",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 2,
        "maxn": 7,
    },
    {
        "name": "cbow_d300_ws10_e10_mc2",
        "model": "cbow",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
    {
        "name": "sg_d300_ws10_e10_mc2_lr01",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
        "lr": 0.1,
    },
]

# Tier 2: Learning rate & loss variations
TIER2_PARAMS = [
    {
        "name": "sg_d300_ws10_e10_mc2_lr025",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
        "lr": 0.25,
    },
    {
        "name": "sg_d300_ws10_e10_mc2_neg10",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
        "neg": 10,
    },
    {
        "name": "sg_d300_ws10_e10_mc2_neg15",
        "model": "skipgram",
        "dim": 300,
        "ws": 10,
        "epoch": 10,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
        "neg": 15,
    },
    {
        "name": "sg_d200_ws10_e15_mc2",
        "model": "skipgram",
        "dim": 200,
        "ws": 10,
        "epoch": 15,
        "minCount": 2,
        "minn": 3,
        "maxn": 6,
    },
]


def get_model_filename(config: dict) -> str:
    """Generate model filename from config."""
    return f"ft_{config['name']}.bin"


def train_model(config: dict, corpus_path: Path, output_dir: Path, verbose: int = 2) -> dict:
    """
    Train a single FastText model with given configuration.

    Returns:
        dict with training results including time and model path
    """
    model_name = config["name"]
    model_path = output_dir / get_model_filename(config)

    # Skip if model already exists
    if model_path.exists():
        print(f"[SKIP] Model already exists: {model_path}")
        return {
            "name": model_name,
            "status": "skipped",
            "model_path": str(model_path),
        }

    print(f"\n{'='*60}")
    print(f"Training model: {model_name}")
    print(f"Configuration: {config}")
    print(f"{'='*60}")

    # Prepare training parameters
    train_params = {
        "input": str(corpus_path),
        "model": config["model"],
        "dim": config["dim"],
        "ws": config["ws"],
        "epoch": config["epoch"],
        "minCount": config["minCount"],
        "minn": config.get("minn", 3),
        "maxn": config.get("maxn", 6),
        "thread": os.cpu_count(),  # Use all CPU cores
        "verbose": verbose,
    }

    # Add optional parameters if specified
    if "lr" in config:
        train_params["lr"] = config["lr"]
    if "neg" in config:
        train_params["neg"] = config["neg"]

    # Train model
    start_time = time.time()
    try:
        model = fasttext.train_unsupervised(**train_params)
        training_time = time.time() - start_time

        # Save model
        model.save_model(str(model_path))

        # Get model info
        vocab_size = len(model.words)

        result = {
            "name": model_name,
            "status": "success",
            "model_path": str(model_path),
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "vocab_size": vocab_size,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n[SUCCESS] Model trained in {training_time/60:.1f} minutes")
        print(f"          Vocabulary size: {vocab_size:,}")
        print(f"          Saved to: {model_path}")

        return result

    except Exception as e:
        training_time = time.time() - start_time
        result = {
            "name": model_name,
            "status": "error",
            "error": str(e),
            "training_time_seconds": training_time,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }
        print(f"\n[ERROR] Training failed: {e}")
        return result


def save_results(results: list, results_path: Path):
    """Save training results to JSON file."""
    # Load existing results if file exists
    existing_results = []
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    # Merge results (update existing or append new)
    existing_names = {r["name"] for r in existing_results}
    for result in results:
        if result["name"] in existing_names:
            # Update existing
            for i, existing in enumerate(existing_results):
                if existing["name"] == result["name"]:
                    existing_results[i] = result
                    break
        else:
            existing_results.append(result)

    # Save
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")


def show_status():
    """Show status of all model configurations."""
    all_params = TIER0_PARAMS + TIER1_PARAMS + TIER2_PARAMS

    # Load existing results
    results_path = RESULTS_DIR / "training_results.json"
    training_results = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                training_results[r["name"]] = r

    print(f"\n{'='*70}")
    print("MODEL TRAINING STATUS")
    print(f"{'='*70}")

    print("\nTier 0 (Quick ~2-3h each):")
    print("-" * 70)
    for i, config in enumerate(TIER0_PARAMS):
        model_path = MODELS_DIR / get_model_filename(config)
        exists = model_path.exists()
        result = training_results.get(config["name"], {})

        if exists:
            size = model_path.stat().st_size / (1024**3)
            time_min = result.get("training_time_minutes", "?")
            if isinstance(time_min, float):
                time_str = f"{time_min:.1f} min"
            else:
                time_str = "unknown"
            print(f"  [{i}] ✅ {config['name']:<40} ({size:.2f} GB, {time_str})")
        else:
            print(f"  [{i}] ⏳ {config['name']:<40} (pending)")

    print("\nTier 1 (Standard ~11h each):")
    print("-" * 70)
    for i, config in enumerate(TIER1_PARAMS, start=len(TIER0_PARAMS)):
        model_path = MODELS_DIR / get_model_filename(config)
        exists = model_path.exists()
        result = training_results.get(config["name"], {})

        if exists:
            size = model_path.stat().st_size / (1024**3)
            time_min = result.get("training_time_minutes", "?")
            if isinstance(time_min, float):
                time_str = f"{time_min:.1f} min"
            else:
                time_str = "unknown"
            print(f"  [{i}] ✅ {config['name']:<40} ({size:.2f} GB, {time_str})")
        else:
            print(f"  [{i}] ⏳ {config['name']:<40} (pending)")

    print("\nTier 2 (Experimental ~11h each):")
    print("-" * 70)
    for i, config in enumerate(TIER2_PARAMS, start=len(TIER0_PARAMS) + len(TIER1_PARAMS)):
        model_path = MODELS_DIR / get_model_filename(config)
        exists = model_path.exists()
        result = training_results.get(config["name"], {})

        if exists:
            size = model_path.stat().st_size / (1024**3)
            time_min = result.get("training_time_minutes", "?")
            if isinstance(time_min, float):
                time_str = f"{time_min:.1f} min"
            else:
                time_str = "unknown"
            print(f"  [{i}] ✅ {config['name']:<40} ({size:.2f} GB, {time_str})")
        else:
            print(f"  [{i}] ⏳ {config['name']:<40} (pending)")

    # Summary
    done = sum(1 for c in all_params if (MODELS_DIR / get_model_filename(c)).exists())
    total = len(all_params)
    print(f"\n{'='*70}")
    print(f"Progress: {done}/{total} models completed ({100*done/total:.0f}%)")
    print(f"{'='*70}")

    if done < total:
        print("\nTo continue training:")
        print("  python scripts/train_optimized.py --tier 0 --limit N  # Quick models (~2-3h each)")
        print("  python scripts/train_optimized.py --tier 1 --limit N  # Standard models (~11h each)")


def main():
    parser = argparse.ArgumentParser(description="Train optimized FastText models for Croatian")
    parser.add_argument("--tier", type=int, choices=[0, 1, 2], help="Train specific tier (0=quick, 1=standard, 2=experimental)")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--config", type=int, help="Train specific config by index (0-based)")
    parser.add_argument("--corpus", type=str, help="Path to corpus file")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level (0-2)")
    parser.add_argument("--limit", type=int, help="Train only N models (excludes already-trained)")
    parser.add_argument("--status", action="store_true", help="Show training status and exit")
    parser.add_argument("--name", type=str, help="Train specific model by name (e.g., 'sg_d300_ws10_e5_mc2')")
    args = parser.parse_args()

    # Handle --status
    if args.status:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        show_status()
        return

    # Determine corpus path
    corpus_path = Path(args.corpus) if args.corpus else CORPUS_PATH
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which configs to train
    all_params = TIER0_PARAMS + TIER1_PARAMS + TIER2_PARAMS

    if args.name is not None:
        # Find config by name
        matching = [c for c in all_params if c["name"] == args.name]
        if not matching:
            print(f"Error: Model '{args.name}' not found. Available models:")
            for c in all_params:
                print(f"  - {c['name']}")
            sys.exit(1)
        configs_to_train = matching
    elif args.config is not None:
        if args.config < 0 or args.config >= len(all_params):
            print(f"Error: Config index {args.config} out of range (0-{len(all_params)-1})")
            sys.exit(1)
        configs_to_train = [all_params[args.config]]
    elif args.tier == 0:
        configs_to_train = TIER0_PARAMS
    elif args.tier == 1:
        configs_to_train = TIER1_PARAMS
    elif args.tier == 2:
        configs_to_train = TIER2_PARAMS
    elif args.all:
        configs_to_train = all_params
    else:
        # Default: show available configs
        print("Available configurations:")
        print("\nTier 0 (Quick ~2-3h each):")
        for i, config in enumerate(TIER0_PARAMS):
            print(f"  [{i}] {config['name']}")
        print("\nTier 1 (Standard ~11h each):")
        for i, config in enumerate(TIER1_PARAMS, start=len(TIER0_PARAMS)):
            print(f"  [{i}] {config['name']}")
        print("\nTier 2 (Experimental ~11h each):")
        for i, config in enumerate(TIER2_PARAMS, start=len(TIER0_PARAMS) + len(TIER1_PARAMS)):
            print(f"  [{i}] {config['name']}")
        print("\nUsage:")
        print("  python train_optimized.py --tier 0    # Train quick models (~5h each)")
        print("  python train_optimized.py --tier 1    # Train standard models (~11h each)")
        print("  python train_optimized.py --tier 2    # Train experimental models")
        print("  python train_optimized.py --all       # Train all models")
        print("  python train_optimized.py --config 0  # Train specific config by index")
        print("  python train_optimized.py --name sg_d300_ws10_e5_mc2  # Train by name")
        return

    # Print training plan
    print(f"\n{'='*60}")
    print("FastText Training - Optimized for Croatian")
    print(f"{'='*60}")
    print(f"Corpus: {corpus_path}")
    print(f"Corpus size: {corpus_path.stat().st_size / (1024**3):.2f} GB")
    print(f"CPU threads: {os.cpu_count()}")
    print(f"Models to train: {len(configs_to_train)}")
    print(f"Output directory: {MODELS_DIR}")

    # Train models
    results = []
    trained_count = 0  # Count of actually trained (not skipped) models

    for i, config in enumerate(configs_to_train):
        # Check if we've hit the limit
        if args.limit and trained_count >= args.limit:
            print(f"\n[LIMIT] Reached limit of {args.limit} trained models. Stopping.")
            print(f"        Use --status to see progress, then run again to continue.")
            break

        print(f"\n[{i+1}/{len(configs_to_train)}] Starting training...")
        result = train_model(config, corpus_path, MODELS_DIR, verbose=args.verbose)
        results.append(result)

        # Only count as trained if it wasn't skipped
        if result["status"] == "success":
            trained_count += 1

        # Save intermediate results
        save_results(results, RESULTS_DIR / "training_results.json")

    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")

    successful = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"Successful: {len(successful)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")

    if successful:
        total_time = sum(r["training_time_seconds"] for r in successful)
        print(f"Total training time: {total_time/60:.1f} minutes")

    if failed:
        print("\nFailed models:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
