#!/usr/bin/env python3
"""
Compare trained FastText models against baseline and generate comparison reports.

Usage:
    python compare_models.py                      # Compare all models
    python compare_models.py --top 5              # Show top 5 models
    python compare_models.py --export-csv         # Export results to CSV
"""

import argparse
import json
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"


def load_evaluation_results() -> list:
    """Load all evaluation results."""
    results_file = RESULTS_DIR / "evaluation_results.json"
    if not results_file.exists():
        print(f"No evaluation results found at: {results_file}")
        print("Run: python evaluate_model.py --all-models")
        return []

    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_results() -> dict:
    """Load baseline evaluation results."""
    baseline_file = RESULTS_DIR / "baseline_evaluation.json"
    if not baseline_file.exists():
        return None

    with open(baseline_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        return results[0] if results else None


def calculate_improvement(model_score: float, baseline_score: float) -> float:
    """Calculate percentage improvement over baseline."""
    if baseline_score == 0:
        return 0
    return ((model_score - baseline_score) / baseline_score) * 100


def compare_models(results: list, baseline: dict = None, top_n: int = None) -> dict:
    """
    Compare models and generate comparison report.

    Returns:
        dict with comparison data
    """
    # Filter successful results
    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        return {"error": "No successful model evaluations found"}

    # Sort by overall score
    sorted_models = sorted(successful, key=lambda x: x.get("overall_score", 0), reverse=True)

    if top_n:
        sorted_models = sorted_models[:top_n]

    comparison = {
        "total_models_evaluated": len(successful),
        "best_model": sorted_models[0]["model"] if sorted_models else None,
        "best_overall_score": sorted_models[0]["overall_score"] if sorted_models else 0,
        "baseline_included": baseline is not None,
        "models": [],
    }

    # Add baseline comparison if available
    if baseline and baseline.get("status") == "success":
        comparison["baseline"] = {
            "model": baseline["model"],
            "overall_score": baseline.get("overall_score", 0),
            "ije_je_mrr": baseline.get("ije_je_errors", {}).get("mrr", 0),
            "diacritic_mrr": baseline.get("diacritic_errors", {}).get("mrr", 0),
            "typo_mrr": baseline.get("general_typos", {}).get("mrr", 0),
            "similarity_corr": baseline.get("semantic_similarity", {}).get("correlation_with_expected", 0),
            "morphological_coverage": baseline.get("morphological_coverage", {}).get("mean_coverage", 0),
        }

    # Process each model
    for model_result in sorted_models:
        model_data = {
            "model": model_result["model"],
            "overall_score": model_result.get("overall_score", 0),
            "metrics": {
                "ije_je_mrr": model_result.get("ije_je_errors", {}).get("mrr", 0),
                "ije_je_recall": model_result.get("ije_je_errors", {}).get("recall_at_10", 0),
                "diacritic_mrr": model_result.get("diacritic_errors", {}).get("mrr", 0),
                "diacritic_recall": model_result.get("diacritic_errors", {}).get("recall_at_10", 0),
                "typo_mrr": model_result.get("general_typos", {}).get("mrr", 0),
                "typo_recall": model_result.get("general_typos", {}).get("recall_at_10", 0),
                "similarity_corr": model_result.get("semantic_similarity", {}).get("correlation_with_expected", 0),
                "morphological_coverage": model_result.get("morphological_coverage", {}).get("mean_coverage", 0),
            },
        }

        # Calculate improvement over baseline if available
        if baseline and baseline.get("status") == "success":
            baseline_score = baseline.get("overall_score", 0)
            model_data["improvement_over_baseline"] = calculate_improvement(
                model_data["overall_score"], baseline_score
            )

        comparison["models"].append(model_data)

    return comparison


def print_comparison_table(comparison: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*100}")
    print("MODEL COMPARISON REPORT")
    print(f"{'='*100}")
    print(f"Total models evaluated: {comparison['total_models_evaluated']}")

    if comparison.get("baseline"):
        print(f"\nBASELINE (Facebook cc.hr.300):")
        b = comparison["baseline"]
        print(f"  Overall Score: {b['overall_score']:.3f}")
        print(f"  ije/je MRR: {b['ije_je_mrr']:.3f}")
        print(f"  Diacritic MRR: {b['diacritic_mrr']:.3f}")
        print(f"  Typo MRR: {b['typo_mrr']:.3f}")

    print(f"\n{'Model':<35} {'Overall':<10} {'ije/je':<10} {'Diacrit':<10} {'Typos':<10} {'Sim':<10} {'Morph':<10}", end="")
    if comparison.get("baseline"):
        print(f" {'vs Base':<10}")
    else:
        print()

    print("-" * 110)

    for model_data in comparison["models"]:
        model_name = model_data["model"][:33]
        m = model_data["metrics"]
        overall = model_data["overall_score"]

        print(f"{model_name:<35} {overall:<10.3f} {m['ije_je_mrr']:<10.3f} {m['diacritic_mrr']:<10.3f} "
              f"{m['typo_mrr']:<10.3f} {m['similarity_corr']:<10.3f} {m['morphological_coverage']:<10.3f}", end="")

        if "improvement_over_baseline" in model_data:
            imp = model_data["improvement_over_baseline"]
            sign = "+" if imp > 0 else ""
            print(f" {sign}{imp:<9.1f}%")
        else:
            print()

    # Best model summary
    if comparison["models"]:
        best = comparison["models"][0]
        print(f"\n{'='*100}")
        print(f"BEST MODEL: {best['model']}")
        print(f"Overall Score: {best['overall_score']:.3f}")

        if "improvement_over_baseline" in best:
            imp = best["improvement_over_baseline"]
            if imp > 0:
                print(f"Improvement over baseline: +{imp:.1f}%")
            elif imp < 0:
                print(f"Compared to baseline: {imp:.1f}%")
            else:
                print("Performance equal to baseline")

        print(f"\nPer-category performance:")
        m = best["metrics"]
        print(f"  - ije/je errors:     MRR={m['ije_je_mrr']:.3f}, Recall@10={m['ije_je_recall']:.3f}")
        print(f"  - Diacritic errors:  MRR={m['diacritic_mrr']:.3f}, Recall@10={m['diacritic_recall']:.3f}")
        print(f"  - General typos:     MRR={m['typo_mrr']:.3f}, Recall@10={m['typo_recall']:.3f}")
        print(f"  - Semantic sim:      Correlation={m['similarity_corr']:.3f}")
        print(f"  - Morphological:     Coverage={m['morphological_coverage']:.3f}")
        print(f"{'='*100}")


def export_to_csv(comparison: dict, output_path: Path):
    """Export comparison results to CSV."""
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Model", "Overall Score", "ije/je MRR", "ije/je Recall@10",
                  "Diacritic MRR", "Diacritic Recall@10", "Typo MRR", "Typo Recall@10",
                  "Similarity Corr", "Morph Coverage"]
        if comparison.get("baseline"):
            header.append("vs Baseline %")
        writer.writerow(header)

        # Baseline row if available
        if comparison.get("baseline"):
            b = comparison["baseline"]
            writer.writerow([
                "BASELINE: " + b["model"],
                b["overall_score"],
                b["ije_je_mrr"], "-",
                b["diacritic_mrr"], "-",
                b["typo_mrr"], "-",
                b["similarity_corr"],
                b["morphological_coverage"],
                "0%"
            ])

        # Model rows
        for model_data in comparison["models"]:
            m = model_data["metrics"]
            row = [
                model_data["model"],
                model_data["overall_score"],
                m["ije_je_mrr"], m["ije_je_recall"],
                m["diacritic_mrr"], m["diacritic_recall"],
                m["typo_mrr"], m["typo_recall"],
                m["similarity_corr"],
                m["morphological_coverage"],
            ]
            if "improvement_over_baseline" in model_data:
                row.append(f"{model_data['improvement_over_baseline']:.1f}%")
            writer.writerow(row)

    print(f"\nCSV exported to: {output_path}")


def generate_recommendations(comparison: dict) -> list:
    """Generate recommendations based on comparison results."""
    recommendations = []

    if not comparison.get("models"):
        return ["No models to analyze"]

    best = comparison["models"][0]
    m = best["metrics"]

    # Analyze weaknesses
    if m["diacritic_mrr"] < 0.3:
        recommendations.append(
            "Diacritic errors (č/ć/š/ž) remain challenging. Consider post-processing rules "
            "or combining FastText with edit-distance for single-character errors."
        )

    if m["ije_je_mrr"] < 0.5:
        recommendations.append(
            "ije/je error handling could be improved. Consider increasing training epochs "
            "or using a larger context window."
        )

    if m["morphological_coverage"] < 0.5:
        recommendations.append(
            "Morphological coverage is low. Consider reducing minCount parameter "
            "to include more rare word forms."
        )

    # General recommendations
    if best.get("improvement_over_baseline", 0) < 0:
        recommendations.append(
            "Custom models underperform baseline. Consider:\n"
            "  1. Using larger corpus if available\n"
            "  2. Fine-tuning on domain-specific text\n"
            "  3. Using baseline for general vocabulary, custom model for domain terms"
        )
    elif best.get("improvement_over_baseline", 0) > 10:
        recommendations.append(
            "Custom model significantly outperforms baseline. "
            "This model is well-suited for the target domain."
        )

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Compare FastText models")
    parser.add_argument("--top", type=int, help="Show only top N models")
    parser.add_argument("--export-csv", action="store_true", help="Export to CSV")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    # Load results
    results = load_evaluation_results()
    if not results:
        sys.exit(1)

    baseline = load_baseline_results()

    # Compare
    comparison = compare_models(results, baseline, args.top)

    # Print table
    print_comparison_table(comparison)

    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 50)
    recommendations = generate_recommendations(comparison)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Save comparison results
    output_path = Path(args.output) if args.output else RESULTS_DIR / "comparison_report.json"
    comparison["recommendations"] = recommendations

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to: {output_path}")

    # Export CSV if requested
    if args.export_csv:
        csv_path = RESULTS_DIR / "comparison_report.csv"
        export_to_csv(comparison, csv_path)


if __name__ == "__main__":
    main()
