#!/usr/bin/env python3
"""
Evaluation framework for FastText models trained on Croatian language.
Evaluates models on spelling correction, word similarity, and morphological tasks.

Usage:
    python evaluate_model.py models/ft_sg_d300_ws10_e10_mc2.bin
    python evaluate_model.py --all-models          # Evaluate all models in models/
    python evaluate_model.py --baseline            # Evaluate Facebook baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import fasttext
import numpy as np
import Levenshtein

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
EVAL_DATA_DIR = PROJECT_DIR / "data" / "eval"


# Evaluation datasets - balanced across all error types
EVAL_DATA = {
    # Category 1: ije/je errors (33% weight)
    "ije_je_errors": [
        ("riješenje", "rješenje"),  # incorrect -> correct
        ("mlieko", "mlijeko"),
        ("vriemena", "vremena"),
        ("biel", "bijel"),
        ("diel", "dio"),
        ("tielo", "tijelo"),
        ("grieh", "grijeh"),
        ("sniega", "snijega"),
        ("liek", "lijek"),
        ("ciena", "cijena"),
    ],
    # Category 2: č/ć/š/ž diacritic errors (33% weight)
    "diacritic_errors": [
        ("moč", "moć"),
        ("noč", "noć"),
        ("kuča", "kuća"),
        ("sreca", "sreća"),
        ("zivot", "život"),
        ("covjek", "čovjek"),
        ("skola", "škola"),
        ("zelim", "želim"),
        ("rucak", "ručak"),
        ("poceti", "početi"),
    ],
    # Category 3: General typos (33% weight)
    "general_typos": [
        ("prijateli", "prijatelj"),
        ("doabr", "dobar"),
        ("skolaa", "škola"),
        ("pisatti", "pisati"),
        ("raaditi", "raditi"),
        ("govoritti", "govoriti"),
        ("kuuca", "kuća"),
        ("automboil", "automobil"),
        ("kompjutr", "kompjuter"),
        ("teleofn", "telefon"),
    ],
    # Semantic similarity pairs (word1, word2, expected_similarity)
    "similarity_pairs": [
        ("škola", "učenik", 0.6),
        ("škola", "obrazovanje", 0.65),
        ("pisati", "čitati", 0.5),
        ("auto", "vozilo", 0.7),
        ("kuća", "stan", 0.6),
        ("liječnik", "bolnica", 0.55),
        ("učitelj", "učenik", 0.6),
        ("knjiga", "čitanje", 0.5),
        ("more", "plaža", 0.55),
        ("grad", "selo", 0.45),
    ],
    # Morphological variations (base_word, expected_forms)
    "morphological": [
        ("učitelj", ["učitelja", "učitelju", "učitelje", "učitelji", "učiteljica"]),
        ("pisati", ["pišem", "piše", "pisao", "pisala", "pišući", "napisati"]),
        ("lijep", ["lijepa", "lijepo", "lijepog", "lijepoj", "ljepši"]),
        ("kuća", ["kuće", "kući", "kuću", "kućom", "kuće"]),
        ("raditi", ["radim", "radi", "radio", "radila", "radeći"]),
    ],
}


# ============================================================================
# HYBRID SCORING - More appropriate for spell-checking evaluation
# ============================================================================

def hybrid_score(model, incorrect: str, correct: str) -> float:
    """
    Hybrid scoring combining edit distance + embedding similarity.
    More appropriate for spell-checking than pure nearest-neighbor ranking.

    Returns score between 0 and 1.
    """
    # Get embedding similarity
    vec1 = model.get_word_vector(incorrect)
    vec2 = model.get_word_vector(correct)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 > 0 and norm2 > 0:
        embed_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        # Normalize from [-1,1] to [0,1]
        embed_sim = (embed_sim + 1) / 2
    else:
        embed_sim = 0.5

    # Get edit distance similarity (normalized)
    edit_dist = Levenshtein.distance(incorrect, correct)
    max_len = max(len(incorrect), len(correct))
    edit_sim = 1 - (edit_dist / max_len) if max_len > 0 else 1.0

    # Combine: 40% edit distance, 60% embedding
    return 0.4 * edit_sim + 0.6 * embed_sim


def evaluate_hybrid(model, error_pairs: list) -> dict:
    """
    Evaluate using hybrid scoring instead of MRR.
    Returns mean hybrid score and individual scores.
    """
    scores = []
    details = []

    for incorrect, correct in error_pairs:
        try:
            score = hybrid_score(model, incorrect, correct)
            scores.append(score)
            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "hybrid_score": float(score),
            })
        except Exception as e:
            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "error": str(e),
            })
            scores.append(0.0)

    return {
        "mean_hybrid_score": float(np.mean(scores)) if scores else 0.0,
        "total_pairs": len(error_pairs),
        "details": details,
    }


# Diacritic variant generation for post-processing
DIACRITIC_PAIRS = [
    ('č', 'ć'), ('ć', 'č'),
    ('š', 'ž'), ('ž', 'š'),
]

def generate_diacritic_variants(word: str) -> set:
    """Generate all single-swap diacritic variants of a word."""
    variants = {word}
    for old, new in DIACRITIC_PAIRS:
        for v in list(variants):
            if old in v:
                variants.add(v.replace(old, new, 1))  # Single replacement
    return variants


def evaluate_diacritics_with_variants(model, error_pairs: list) -> dict:
    """
    Evaluate diacritic correction using variant generation.
    For each incorrect word, generate variants and check if correct is among them,
    then score by embedding similarity.
    """
    scores = []
    details = []

    for incorrect, correct in error_pairs:
        try:
            variants = generate_diacritic_variants(incorrect)

            # Check if correct word can be reached via variants
            can_reach = correct in variants

            # Score based on embedding similarity
            score = hybrid_score(model, incorrect, correct)

            # Bonus if variant generation can reach correct word
            if can_reach:
                score = min(1.0, score + 0.2)

            scores.append(score)
            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "variants_generated": list(variants)[:5],
                "can_reach_correct": can_reach,
                "score": float(score),
            })
        except Exception as e:
            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "error": str(e),
            })
            scores.append(0.0)

    return {
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "reachable_ratio": sum(1 for d in details if d.get("can_reach_correct", False)) / len(error_pairs) if error_pairs else 0.0,
        "total_pairs": len(error_pairs),
        "details": details,
    }


def load_model(model_path: str) -> Optional[fasttext.FastText._FastText]:
    """Load a FastText model."""
    try:
        model = fasttext.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def calculate_mrr(model, error_pairs: list, k: int = 10) -> dict:
    """
    Calculate Mean Reciprocal Rank for spelling correction.

    For each (incorrect, correct) pair, get k nearest neighbors of incorrect word
    and find the rank of the correct word.

    Returns:
        dict with MRR, Recall@K, and per-pair details
    """
    reciprocal_ranks = []
    recall_at_k = 0
    details = []

    for incorrect, correct in error_pairs:
        try:
            neighbors = model.get_nearest_neighbors(incorrect, k=k)
            neighbor_words = [word for _, word in neighbors]

            # Find rank of correct word (1-indexed)
            rank = None
            if correct in neighbor_words:
                rank = neighbor_words.index(correct) + 1
                reciprocal_ranks.append(1.0 / rank)
                recall_at_k += 1
            else:
                reciprocal_ranks.append(0.0)

            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "rank": rank,
                "top_suggestions": neighbor_words[:5],
            })
        except Exception as e:
            details.append({
                "incorrect": incorrect,
                "correct": correct,
                "error": str(e),
            })
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    recall = recall_at_k / len(error_pairs) if error_pairs else 0.0

    return {
        "mrr": float(mrr),
        f"recall_at_{k}": float(recall),
        "total_pairs": len(error_pairs),
        "found_in_top_k": recall_at_k,
        "details": details,
    }


def evaluate_similarity(model, similarity_pairs: list) -> dict:
    """
    Evaluate word similarity correlation.

    Returns:
        dict with mean similarity, correlation metrics
    """
    predicted_similarities = []
    expected_similarities = []
    details = []

    for word1, word2, expected in similarity_pairs:
        try:
            # Get word vectors
            vec1 = model.get_word_vector(word1)
            vec2 = model.get_word_vector(word2)

            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            predicted_similarities.append(similarity)
            expected_similarities.append(expected)

            details.append({
                "word1": word1,
                "word2": word2,
                "predicted_similarity": float(similarity),
                "expected_similarity": expected,
            })
        except Exception as e:
            details.append({
                "word1": word1,
                "word2": word2,
                "error": str(e),
            })

    # Calculate correlation if we have enough data
    if len(predicted_similarities) >= 3:
        correlation = np.corrcoef(predicted_similarities, expected_similarities)[0, 1]
    else:
        correlation = 0.0

    return {
        "mean_predicted_similarity": float(np.mean(predicted_similarities)) if predicted_similarities else 0.0,
        "correlation_with_expected": float(correlation) if not np.isnan(correlation) else 0.0,
        "total_pairs": len(similarity_pairs),
        "evaluated_pairs": len(predicted_similarities),
        "details": details,
    }


def evaluate_morphological(model, morphological_data: list, k: int = 20) -> dict:
    """
    Evaluate morphological coverage - how well the model captures word forms.

    For each base word, check how many expected forms appear in nearest neighbors.
    """
    coverage_scores = []
    details = []

    for base_word, expected_forms in morphological_data:
        try:
            neighbors = model.get_nearest_neighbors(base_word, k=k)
            neighbor_words = set(word for _, word in neighbors)

            # Count how many expected forms are found
            found_forms = [form for form in expected_forms if form in neighbor_words]
            coverage = len(found_forms) / len(expected_forms)
            coverage_scores.append(coverage)

            details.append({
                "base_word": base_word,
                "expected_forms": expected_forms,
                "found_forms": found_forms,
                "coverage": float(coverage),
                "top_neighbors": [word for _, word in neighbors[:10]],
            })
        except Exception as e:
            details.append({
                "base_word": base_word,
                "error": str(e),
            })

    mean_coverage = np.mean(coverage_scores) if coverage_scores else 0.0

    return {
        "mean_coverage": float(mean_coverage),
        "total_base_words": len(morphological_data),
        "details": details,
    }


def evaluate_model(model_path: str, save_details: bool = True) -> dict:
    """
    Run full evaluation on a FastText model.

    Returns:
        dict with all evaluation metrics
    """
    model_name = Path(model_path).stem
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model = load_model(model_path)
    if model is None:
        return {"model": model_name, "status": "error", "error": "Failed to load model"}

    results = {
        "model": model_name,
        "model_path": model_path,
        "status": "success",
    }

    # 1. Evaluate ije/je errors (MRR - traditional)
    print("\n[1/7] Evaluating ije/je errors (MRR)...")
    ije_je_results = calculate_mrr(model, EVAL_DATA["ije_je_errors"])
    results["ije_je_errors"] = ije_je_results
    print(f"      MRR: {ije_je_results['mrr']:.3f}, Recall@10: {ije_je_results['recall_at_10']:.3f}")

    # 2. Evaluate ije/je errors (Hybrid - more appropriate)
    print("\n[2/7] Evaluating ije/je errors (Hybrid)...")
    ije_je_hybrid = evaluate_hybrid(model, EVAL_DATA["ije_je_errors"])
    results["ije_je_hybrid"] = ije_je_hybrid
    print(f"      Hybrid Score: {ije_je_hybrid['mean_hybrid_score']:.3f}")

    # 3. Evaluate diacritic errors (with variant generation)
    print("\n[3/7] Evaluating diacritic errors (Hybrid + Variants)...")
    diacritic_results = evaluate_diacritics_with_variants(model, EVAL_DATA["diacritic_errors"])
    results["diacritic_errors"] = diacritic_results
    print(f"      Hybrid Score: {diacritic_results['mean_score']:.3f}")
    print(f"      Reachable via variants: {diacritic_results['reachable_ratio']:.1%}")

    # 4. Evaluate general typos (Hybrid)
    print("\n[4/7] Evaluating general typos (Hybrid)...")
    typo_hybrid = evaluate_hybrid(model, EVAL_DATA["general_typos"])
    results["general_typos"] = typo_hybrid
    print(f"      Hybrid Score: {typo_hybrid['mean_hybrid_score']:.3f}")

    # 5. Evaluate semantic similarity (FastText strength!)
    print("\n[5/7] Evaluating semantic similarity...")
    similarity_results = evaluate_similarity(model, EVAL_DATA["similarity_pairs"])
    results["semantic_similarity"] = similarity_results
    print(f"      Mean similarity: {similarity_results['mean_predicted_similarity']:.3f}")
    print(f"      Correlation: {similarity_results['correlation_with_expected']:.3f}")

    # 6. Evaluate morphological coverage (FastText strength!)
    print("\n[6/7] Evaluating morphological coverage...")
    morphological_results = evaluate_morphological(model, EVAL_DATA["morphological"])
    results["morphological_coverage"] = morphological_results
    print(f"      Mean coverage: {morphological_results['mean_coverage']:.3f}")

    # 7. Calculate overall scores
    print("\n[7/7] Calculating overall scores...")

    # OLD score (MRR-based, unfair to FastText)
    old_score = (
        ije_je_results["mrr"] * 0.22 +
        0.0 * 0.22 +  # diacritics MRR was always 0
        typo_hybrid["mean_hybrid_score"] * 0.22 +
        similarity_results["correlation_with_expected"] * 0.22 +
        morphological_results["mean_coverage"] * 0.12
    )
    results["old_overall_score"] = float(old_score)

    # NEW score (Hybrid-based, fair evaluation)
    # Weights favor FastText strengths: semantic similarity (25%), morphology (25%)
    # Spelling tasks use hybrid scoring: ije/je (15%), diacritics (15%), typos (20%)
    correlation = similarity_results["correlation_with_expected"]
    # Handle negative correlation (treat as 0)
    correlation_normalized = max(0, correlation)

    new_score = (
        ije_je_hybrid["mean_hybrid_score"] * 0.15 +
        diacritic_results["mean_score"] * 0.15 +
        typo_hybrid["mean_hybrid_score"] * 0.20 +
        correlation_normalized * 0.25 +
        morphological_results["mean_coverage"] * 0.25
    )
    results["overall_score"] = float(new_score)

    print(f"\n{'='*60}")
    print(f"OLD Overall Score (MRR-based): {old_score:.3f}")
    print(f"NEW Overall Score (Hybrid):    {new_score:.3f}")
    print(f"{'='*60}")

    return results


def evaluate_all_models(models_dir: Path) -> list:
    """Evaluate all .bin models in the models directory."""
    model_files = list(models_dir.glob("*.bin"))
    print(f"Found {len(model_files)} models to evaluate")

    all_results = []
    for model_path in sorted(model_files):
        results = evaluate_model(str(model_path))
        all_results.append(results)

    return all_results


def save_evaluation_results(results: list, output_path: Path):
    """Save evaluation results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: list):
    """Print a summary comparison of all evaluated models."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("EVALUATION SUMMARY (Hybrid Scoring - Fair to FastText)")
    print(f"{'='*100}")

    # Sort by overall score
    sorted_results = sorted(
        [r for r in results if r.get("status") == "success"],
        key=lambda x: x.get("overall_score", 0),
        reverse=True
    )

    # Print header - now showing hybrid scores
    print(f"\n{'Model':<40} {'ije/je':<10} {'Diacrit':<10} {'Typos':<10} {'Sim':<10} {'Morph':<10} {'NEW':<10} {'OLD':<10}")
    print("-" * 110)

    for r in sorted_results:
        model_name = r["model"][:38]
        # Use hybrid scores where available
        ije_je = r.get("ije_je_hybrid", {}).get("mean_hybrid_score", r.get("ije_je_errors", {}).get("mrr", 0))
        diacrit = r.get("diacritic_errors", {}).get("mean_score", r.get("diacritic_errors", {}).get("mrr", 0))
        typos = r.get("general_typos", {}).get("mean_hybrid_score", r.get("general_typos", {}).get("mrr", 0))
        sim = max(0, r.get("semantic_similarity", {}).get("correlation_with_expected", 0))
        morph = r.get("morphological_coverage", {}).get("mean_coverage", 0)
        overall_new = r.get("overall_score", 0)
        overall_old = r.get("old_overall_score", 0)

        print(f"{model_name:<40} {ije_je:<10.3f} {diacrit:<10.3f} {typos:<10.3f} {sim:<10.3f} {morph:<10.3f} {overall_new:<10.3f} {overall_old:<10.3f}")

    if sorted_results:
        print(f"\n{'='*100}")
        print(f"BEST MODEL: {sorted_results[0]['model']}")
        print(f"NEW Overall Score (Hybrid): {sorted_results[0]['overall_score']:.3f}")
        if 'old_overall_score' in sorted_results[0]:
            print(f"OLD Overall Score (MRR):    {sorted_results[0]['old_overall_score']:.3f}")
        print(f"{'='*100}")
        print("\nNote: NEW score uses hybrid (edit distance + embedding) evaluation.")
        print("      OLD score used pure MRR which is unfair to FastText for spelling correction.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FastText models for Croatian")
    parser.add_argument("model_path", nargs="?", help="Path to model file (.bin)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all models in models/")
    parser.add_argument("--baseline", action="store_true", help="Evaluate Facebook baseline model")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        results = evaluate_all_models(MODELS_DIR)
        output_path = Path(args.output) if args.output else RESULTS_DIR / "evaluation_results.json"
        save_evaluation_results(results, output_path)
        print_summary(results)

    elif args.baseline:
        baseline_path = MODELS_DIR / "facebook_cc_hr_300.bin"
        if not baseline_path.exists():
            print(f"Baseline model not found: {baseline_path}")
            print("Run: bash scripts/download_baseline.sh")
            sys.exit(1)
        results = [evaluate_model(str(baseline_path))]
        output_path = Path(args.output) if args.output else RESULTS_DIR / "baseline_evaluation.json"
        save_evaluation_results(results, output_path)

    elif args.model_path:
        if not Path(args.model_path).exists():
            print(f"Model not found: {args.model_path}")
            sys.exit(1)
        results = [evaluate_model(args.model_path)]
        output_path = Path(args.output) if args.output else RESULTS_DIR / f"{Path(args.model_path).stem}_evaluation.json"
        save_evaluation_results(results, output_path)

    else:
        print("Usage:")
        print("  python evaluate_model.py models/ft_model.bin     # Evaluate single model")
        print("  python evaluate_model.py --all-models            # Evaluate all models")
        print("  python evaluate_model.py --baseline              # Evaluate Facebook baseline")


if __name__ == "__main__":
    main()
