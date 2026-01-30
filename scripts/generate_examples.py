#!/usr/bin/env python3
"""
Generate qualitative example outputs for FastText model comparison.

Creates tables showing nearest neighbors for test words across all models,
similar to Seminar 2 format. Also includes word analogies and similarity scores.

IMPORTANT: Loads models one at a time to avoid memory issues.

Usage:
    python scripts/generate_examples.py [--output-format markdown|json|both]
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import fasttext
except ImportError:
    print("Error: fasttext not installed. Run: pip install fasttext")
    sys.exit(1)


# Test words for evaluation - highlighting FastText STRENGTHS

# 1. Semantic clustering examples (FastText strength)
TEST_WORDS_CORRECT = [
    "škola",       # Education cluster
    "Zagreb",      # Croatian cities cluster
    "ponedjeljak", # Days of week cluster
    "učitelj",     # Morphological forms (7 cases)
    "računalo",    # Tech/domain terms
]

# 2. ije/je errors - FastText EXCELS at these
TEST_WORDS_MISSPELLED = [
    ("riješenje", "rješenje"),   # ije/je - FastText gets #1!
    ("mlieko", "mlijeko"),       # ije/je - FastText gets #1!
    ("biel", "bijel"),           # ije/je - FastText gets #1!
    ("moč", "moć"),              # Diacritic - acknowledged limitation
]

# Extended test set
EXTENDED_CORRECT = {
    "imenice": ["škola", "kuća", "grad"],
    "glagoli": ["pisati", "čitati", "raditi"],
    "dijakritici": ["moć", "noć", "učitelj"],
}

EXTENDED_MISSPELLED = {
    "ije_je": [
        ("riješenje", "rješenje"),
        ("mlieko", "mlijeko"),
        ("vriemena", "vremena"),
        ("biel", "bijel"),
    ],
    "dijakritici": [
        ("moč", "moć"),
        ("noč", "noć"),
        ("kuča", "kuća"),
        ("sreča", "sreća"),
    ],
    "tipfeleri": [
        ("prijateli", "prijatelj"),
        ("doabr", "dobar"),
        ("skolaa", "škola"),
    ],
}

# Word pairs for similarity comparison - showing semantic understanding
SIMILARITY_PAIRS = [
    ("pisati", "čitati"),        # High - related verbs
    ("auto", "vozilo"),          # High - synonyms
    ("škola", "učenik"),         # High - related concepts
    ("Zagreb", "Split"),         # High - both Croatian cities
    ("ponedjeljak", "utorak"),   # High - both days of week
    ("učitelj", "profesor"),     # High - related professions
    ("škola", "banana"),         # Low - unrelated (baseline)
]

# Word analogy pairs (A is to B as C is to ?) - FastText gets some right!
ANALOGIES = [
    ("kralj", "kraljica", "princ", "princeza"),   # Gender - models get this!
    ("Zagreb", "Hrvatska", "Rim", "Italija"),     # Capital:Country
    ("ponedjeljak", "utorak", "srijeda", "četvrtak"),  # Day sequence
    ("učitelj", "učiteljica", "student", "studentica"), # Gender profession
]


def is_clean_word(word: str) -> bool:
    """
    Check if a word is "clean" (no attached punctuation).

    Returns True for:
    - Pure alphabetic words: "škola", "pisati"
    - Compound words with hyphen in middle: "E-škola", "auto-škola"

    Returns False for:
    - Words with trailing punctuation: "škola,", "pisati."
    - Words with leading punctuation: "-škola", "'pisati"
    - Words with embedded punctuation: "škola,od", "osn.škola"
    """
    if not word:
        return False

    # Must start and end with a letter
    if not word[0].isalpha() or not word[-1].isalpha():
        return False

    # Check for invalid punctuation (anything except hyphen in middle)
    for i, ch in enumerate(word):
        if not ch.isalpha():
            # Allow hyphen only if surrounded by letters
            if ch == '-':
                if i == 0 or i == len(word) - 1:
                    return False
                if not word[i-1].isalpha() or not word[i+1].isalpha():
                    return False
            else:
                return False

    return True


def get_model_files(models_dir: Path) -> List[Path]:
    """Get list of model files in preferred order."""
    preferred_order = [
        "ft_sg_d300_ws10_e15_mc2.bin",
        "ft_sg_d300_ws10_e10_mc2.bin",
        "ft_cbow_d300_ws10_e10_mc2.bin",
        "ft_sg_d200_ws10_e3_mc2.bin",
        "ft_sg_d200_ws5_e3_mc2.bin",
        "ft_sg_d100_ws5_e3_mc2.bin",
        "ft_cbow_d200_ws5_e3_mc2.bin",
        "ft_sg_d300_ws10_e10_mc2_minn2_maxn7.bin",
        "facebook_cc_hr_300.bin",
    ]

    model_files = []
    for filename in preferred_order:
        filepath = models_dir / filename
        if filepath.exists():
            model_files.append(filepath)

    return model_files


def get_short_name(filename: str) -> str:
    """Get abbreviated model name for table headers."""
    name = filename.replace(".bin", "").replace("ft_", "")

    replacements = {
        "sg_d300_ws10_e15_mc2": "SG-300-E15",
        "sg_d300_ws10_e10_mc2": "SG-300-E10",
        "cbow_d300_ws10_e10_mc2": "CB-300",
        "sg_d200_ws10_e3_mc2": "SG-200-W10",
        "sg_d200_ws5_e3_mc2": "SG-200-W5",
        "sg_d100_ws5_e3_mc2": "SG-100",
        "cbow_d200_ws5_e3_mc2": "CB-200",
        "sg_d300_ws10_e10_mc2_minn2_maxn7": "SG-300-N7",
        "facebook_cc_hr_300": "FB-300",
    }

    return replacements.get(name, name[:12])


def get_nearest_neighbors(model, word: str, k: int = 10, filter_clean: bool = True) -> List[Tuple[str, float]]:
    """Get k nearest neighbors for a word.

    Args:
        model: FastText model
        word: Query word
        k: Number of neighbors to return
        filter_clean: If True, only return "clean" words (no punctuation attached)
    """
    try:
        # Fetch more neighbors than needed to account for filtering
        fetch_k = k * 5 if filter_clean else k
        neighbors = model.get_nearest_neighbors(word, fetch_k)

        results = []
        for score, w in neighbors:
            if filter_clean and not is_clean_word(w):
                continue
            results.append((w, round(float(score), 4)))
            if len(results) >= k:
                break

        return results
    except Exception:
        return []


def cosine_similarity(model, word1: str, word2: str) -> float:
    """Calculate cosine similarity between two words."""
    try:
        vec1 = model.get_word_vector(word1)
        vec2 = model.get_word_vector(word2)
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return round(float(sim), 4)
    except Exception:
        return 0.0


def word_analogy(model, a: str, b: str, c: str, k: int = 5) -> List[Tuple[str, float]]:
    """Compute word analogy: A is to B as C is to ?"""
    try:
        vec_a = model.get_word_vector(a)
        vec_b = model.get_word_vector(b)
        vec_c = model.get_word_vector(c)

        result_vec = vec_b - vec_a + vec_c
        result_vec = result_vec / np.linalg.norm(result_vec)

        # Get many neighbors and filter
        neighbors = model.get_nearest_neighbors(c, k * 5)

        scored = []
        for score, word in neighbors:
            if word.lower() not in [a.lower(), b.lower(), c.lower()]:
                vec_w = model.get_word_vector(word)
                vec_w_norm = vec_w / np.linalg.norm(vec_w)
                sim = np.dot(result_vec, vec_w_norm)
                scored.append((word, round(float(sim), 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
    except Exception:
        return []


def collect_data_for_model(model_path: Path, words: List[str], misspelled: List[Tuple[str, str]],
                           similarity_pairs: List[Tuple[str, str]], analogies: List[Tuple[str, str, str, str]]) -> dict:
    """Load model, collect all data, then unload to save memory."""

    model_name = model_path.name
    print(f"  Loading {model_name}...")

    try:
        model = fasttext.load_model(str(model_path))
    except Exception as e:
        print(f"    Error loading: {e}")
        return None

    results = {
        "model": model_name,
        "short_name": get_short_name(model_name),
        "neighbors": {},
        "spelling": {},
        "similarity": {},
        "analogies": {},
    }

    # Collect nearest neighbors for correct words
    print(f"    Getting neighbors for {len(words)} correct words...")
    for word in words:
        results["neighbors"][word] = get_nearest_neighbors(model, word, 10)

    # Collect data for misspelled words
    print(f"    Checking {len(misspelled)} misspelled words...")
    for incorrect, correct in misspelled:
        neighbors = get_nearest_neighbors(model, incorrect, 10)
        neighbor_words = [w for w, s in neighbors]

        rank = -1
        if correct in neighbor_words:
            rank = neighbor_words.index(correct) + 1

        results["spelling"][incorrect] = {
            "correct": correct,
            "neighbors": neighbors,
            "rank": rank,
            "found": rank > 0,
        }

    # Collect similarity scores
    print(f"    Computing {len(similarity_pairs)} similarity scores...")
    for word1, word2 in similarity_pairs:
        results["similarity"][f"{word1}-{word2}"] = cosine_similarity(model, word1, word2)

    # Collect analogy results
    print(f"    Testing {len(analogies)} analogies...")
    for a, b, c, expected in analogies:
        analogy_results = word_analogy(model, a, b, c, 3)
        results["analogies"][f"{a}:{b}::{c}:?"] = {
            "expected": expected,
            "results": analogy_results,
            "top_word": analogy_results[0][0] if analogy_results else "-",
        }

    # Unload model to free memory
    del model
    gc.collect()
    print(f"    Done with {model_name}")

    return results


def generate_markdown_tables(all_results: List[dict], output_dir: Path) -> str:
    """Generate markdown report from collected results."""

    lines = []
    lines.append("# Primjeri izlaza modela\n")
    lines.append("Kvalitativna usporedba nearest neighbors za testne riječi.\n")
    lines.append("---\n")

    model_names = [r["short_name"] for r in all_results]

    # Section 1: Correctly written words
    lines.append("## 1. Ispravno napisane riječi\n")

    # Get all correct words from first result
    correct_words = list(all_results[0]["neighbors"].keys())

    for word in correct_words:
        lines.append(f"### Riječ: **{word}**\n")

        # Header
        header = "| # | " + " | ".join(model_names) + " |"
        separator = "|---" + "|---" * len(model_names) + "|"
        lines.append(header)
        lines.append(separator)

        # Rows (top 10 neighbors)
        for i in range(10):
            row = [f"{i+1}"]
            for result in all_results:
                neighbors = result["neighbors"].get(word, [])
                if i < len(neighbors):
                    w, s = neighbors[i]
                    row.append(f"{w} ({s:.3f})")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # Section 2: Misspelled words - summary table
    lines.append("---\n")
    lines.append("## 2. Pogrešno napisane riječi\n")
    lines.append("Tablica pokazuje pojavljuje li se ispravna riječ u top-10 susjeda (✓ = da, s rangom).\n")

    # Get misspelled words from first result
    misspelled_words = list(all_results[0]["spelling"].keys())

    # Header
    header = "| Pogrešno → Ispravno | " + " | ".join(model_names) + " |"
    separator = "|---" + "|---" * len(model_names) + "|"
    lines.append(header)
    lines.append(separator)

    for incorrect in misspelled_words:
        correct = all_results[0]["spelling"][incorrect]["correct"]
        row = [f"{incorrect} → {correct}"]

        for result in all_results:
            data = result["spelling"].get(incorrect, {})
            if data.get("found"):
                row.append(f"✓ #{data['rank']}")
            else:
                top = data.get("neighbors", [])
                if top:
                    row.append(f"✗ ({top[0][0]})")
                else:
                    row.append("✗")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Detailed view for riješenje and moč
    for key_word in ["riješenje", "moč"]:
        if key_word in misspelled_words:
            lines.append(f"### Detaljno: **{key_word}**\n")

            header = "| # | " + " | ".join(model_names) + " |"
            separator = "|---" + "|---" * len(model_names) + "|"
            lines.append(header)
            lines.append(separator)

            for i in range(10):
                row = [f"{i+1}"]
                for result in all_results:
                    neighbors = result["spelling"].get(key_word, {}).get("neighbors", [])
                    if i < len(neighbors):
                        w, s = neighbors[i]
                        # Highlight correct word
                        correct = result["spelling"][key_word]["correct"]
                        if w == correct:
                            row.append(f"**{w}** ({s:.3f})")
                        else:
                            row.append(f"{w} ({s:.3f})")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

    # Section 3: Semantic similarity
    lines.append("---\n")
    lines.append("## 3. Semantička sličnost\n")
    lines.append("Kosinusna sličnost između parova riječi.\n")

    # Get similarity pairs from first result
    sim_pairs = list(all_results[0]["similarity"].keys())

    header = "| Par riječi | " + " | ".join(model_names) + " |"
    separator = "|---" + "|---" * len(model_names) + "|"
    lines.append(header)
    lines.append(separator)

    for pair in sim_pairs:
        row = [pair]
        for result in all_results:
            sim = result["similarity"].get(pair, 0)
            row.append(f"{sim:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Section 4: Word analogies
    lines.append("---\n")
    lines.append("## 4. Analogije riječi\n")
    lines.append("Test: A : B :: C : ? (očekivani odgovor u zagradi)\n")

    # Get analogies from first result
    analogy_keys = list(all_results[0]["analogies"].keys())

    header = "| Analogija | Očekivano | " + " | ".join(model_names) + " |"
    separator = "|---" + "|---" * (len(model_names) + 1) + "|"
    lines.append(header)
    lines.append(separator)

    for key in analogy_keys:
        expected = all_results[0]["analogies"][key]["expected"]
        row = [key, expected]

        for result in all_results:
            top_word = result["analogies"].get(key, {}).get("top_word", "-")
            # Highlight if correct
            if top_word.lower() == expected.lower():
                row.append(f"**{top_word}**")
            else:
                row.append(top_word)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Section 5: Legend
    lines.append("---\n")
    lines.append("## 5. Legenda modela\n")
    lines.append("| Kratica | Puni naziv | Opis |")
    lines.append("|---------|------------|------|")

    descriptions = {
        "SG-300-E15": "Skip-gram, dim=300, epochs=15",
        "SG-300-E10": "Skip-gram, dim=300, epochs=10",
        "CB-300": "CBOW, dim=300, epochs=10",
        "SG-200-W10": "Skip-gram, dim=200, window=10",
        "SG-200-W5": "Skip-gram, dim=200, window=5",
        "SG-100": "Skip-gram, dim=100",
        "CB-200": "CBOW, dim=200",
        "SG-300-N7": "Skip-gram, minn=2, maxn=7",
        "FB-300": "Facebook pre-trained (6.4B tokens)",
    }

    for result in all_results:
        short = result["short_name"]
        full = result["model"].replace(".bin", "")
        desc = descriptions.get(short, "-")
        lines.append(f"| {short} | {full} | {desc} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate model example outputs")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Output directory"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Models directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of models (0 = all)"
    )
    args = parser.parse_args()

    # Get model files
    model_files = get_model_files(args.models_dir)

    if not model_files:
        print(f"Error: No models found in {args.models_dir}")
        sys.exit(1)

    if args.limit > 0:
        model_files = model_files[:args.limit]

    print(f"Found {len(model_files)} models")

    # Prepare test data
    all_words = TEST_WORDS_CORRECT
    all_misspelled = TEST_WORDS_MISSPELLED

    # Collect data from each model (one at a time)
    print("\nCollecting data from models (one at a time to save memory)...\n")
    all_results = []

    for model_path in model_files:
        result = collect_data_for_model(
            model_path,
            all_words,
            all_misspelled,
            SIMILARITY_PAIRS,
            ANALOGIES
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("Error: No results collected")
        sys.exit(1)

    print(f"\nCollected data from {len(all_results)} models")

    # Generate output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format in ["markdown", "both"]:
        print("\nGenerating markdown report...")
        report = generate_markdown_tables(all_results, args.output_dir)

        output_file = args.output_dir / "example_outputs.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved: {output_file}")

    if args.output_format in ["json", "both"]:
        output_file = args.output_dir / "example_outputs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
