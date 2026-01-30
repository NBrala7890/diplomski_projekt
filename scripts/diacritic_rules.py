#!/usr/bin/env python3
"""
Diacritic correction rules for Croatian language.

FastText embeddings cannot distinguish between č/ć, š/ž because they have
nearly identical character n-grams. This module provides rule-based
post-processing to handle diacritic correction.

Usage:
    from diacritic_rules import correct_diacritics, generate_diacritic_variants

    # Generate all variants of a word
    variants = generate_diacritic_variants("moč")
    # Returns: {'moč', 'moć'}

    # Correct diacritics using a FastText model
    corrected = correct_diacritics("moč", model)
    # Returns: "moć" (if model scores it higher)
"""

import numpy as np
from typing import Set, Optional, List, Tuple

# Croatian diacritic pairs that are commonly confused
DIACRITIC_PAIRS = [
    ('č', 'ć'),
    ('ć', 'č'),
    ('š', 'ž'),
    ('ž', 'š'),
    ('dž', 'đ'),
    ('đ', 'dž'),
]

# Common words with diacritic patterns (for dictionary-based correction)
COMMON_DIACRITIC_WORDS = {
    # č vs ć
    'moć': 'moć',
    'moč': 'moć',
    'noć': 'noć',
    'noč': 'noć',
    'kuća': 'kuća',
    'kuča': 'kuća',
    'sreća': 'sreća',
    'sreča': 'sreća',
    'pomoć': 'pomoć',
    'pomoč': 'pomoć',
    'značka': 'značka',
    'značća': 'značka',
    # Add more as needed
}


def generate_diacritic_variants(word: str, max_swaps: int = 2) -> Set[str]:
    """
    Generate all diacritic variants of a word.

    Args:
        word: Input word (possibly with incorrect diacritics)
        max_swaps: Maximum number of character swaps to apply

    Returns:
        Set of all possible diacritic variants
    """
    variants = {word}

    for _ in range(max_swaps):
        new_variants = set()
        for variant in variants:
            for old, new in DIACRITIC_PAIRS:
                if old in variant:
                    # Replace first occurrence
                    new_word = variant.replace(old, new, 1)
                    new_variants.add(new_word)
        variants.update(new_variants)

    return variants


def score_word_by_embedding(word: str, model) -> float:
    """
    Score a word by its embedding properties.

    Uses vector norm as a proxy for word frequency/commonness.
    Words that appear more often in the corpus tend to have
    more "stable" (higher norm) embeddings.

    Args:
        word: Word to score
        model: FastText model

    Returns:
        Score (higher = more likely to be correct)
    """
    try:
        vec = model.get_word_vector(word)
        return float(np.linalg.norm(vec))
    except Exception:
        return 0.0


def correct_diacritics(
    word: str,
    model=None,
    dictionary: Optional[Set[str]] = None,
    use_common_words: bool = True
) -> str:
    """
    Correct diacritics in a Croatian word.

    Strategy:
    1. Check common word dictionary first (fast, accurate)
    2. Generate diacritic variants
    3. Filter by dictionary if available
    4. Score variants by embedding (if model provided)
    5. Return highest-scoring variant

    Args:
        word: Input word (possibly with incorrect diacritics)
        model: Optional FastText model for scoring
        dictionary: Optional set of valid Croatian words
        use_common_words: Whether to check common word list first

    Returns:
        Corrected word (or original if no correction found)
    """
    # Step 1: Check common words dictionary
    if use_common_words and word.lower() in COMMON_DIACRITIC_WORDS:
        return COMMON_DIACRITIC_WORDS[word.lower()]

    # Step 2: Generate variants
    variants = generate_diacritic_variants(word)

    if len(variants) <= 1:
        return word  # No diacritics to swap

    # Step 3: Filter by dictionary if available
    if dictionary:
        valid_variants = {v for v in variants if v in dictionary}
        if valid_variants:
            variants = valid_variants
        # If no valid variants, keep all (may be OOV word)

    # Step 4: Score by embedding if model provided
    if model is not None:
        scored = [(v, score_word_by_embedding(v, model)) for v in variants]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    # Step 5: Return original if no model
    return word


def correct_text(
    text: str,
    model=None,
    dictionary: Optional[Set[str]] = None
) -> str:
    """
    Correct diacritics in a full text.

    Args:
        text: Input text
        model: Optional FastText model for scoring
        dictionary: Optional set of valid Croatian words

    Returns:
        Text with corrected diacritics
    """
    words = text.split()
    corrected_words = []

    for word in words:
        # Preserve punctuation
        prefix = ""
        suffix = ""

        while word and not word[0].isalpha():
            prefix += word[0]
            word = word[1:]

        while word and not word[-1].isalpha():
            suffix = word[-1] + suffix
            word = word[:-1]

        if word:
            corrected = correct_diacritics(word, model, dictionary)
            corrected_words.append(prefix + corrected + suffix)
        else:
            corrected_words.append(prefix + suffix)

    return " ".join(corrected_words)


def evaluate_diacritic_correction(
    test_pairs: List[Tuple[str, str]],
    model=None,
    dictionary: Optional[Set[str]] = None
) -> dict:
    """
    Evaluate diacritic correction accuracy.

    Args:
        test_pairs: List of (incorrect, correct) word pairs
        model: Optional FastText model
        dictionary: Optional word dictionary

    Returns:
        Evaluation results with accuracy and details
    """
    correct = 0
    details = []

    for incorrect, expected in test_pairs:
        predicted = correct_diacritics(incorrect, model, dictionary)
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        details.append({
            "incorrect": incorrect,
            "expected": expected,
            "predicted": predicted,
            "is_correct": is_correct,
        })

    accuracy = correct / len(test_pairs) if test_pairs else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_pairs),
        "details": details,
    }


if __name__ == "__main__":
    # Demo usage
    print("Diacritic Correction Demo")
    print("=" * 40)

    test_words = ["moč", "noč", "kuča", "sreča", "pomoč"]

    print("\nWithout model (common words only):")
    for word in test_words:
        corrected = correct_diacritics(word)
        print(f"  {word} -> {corrected}")

    print("\nVariant generation:")
    for word in test_words[:2]:
        variants = generate_diacritic_variants(word)
        print(f"  {word} -> {variants}")
