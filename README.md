# FastText Word Embeddings for Croatian Spellchecker

**Optimizing FastText models for semantic spelling correction in Croatian language**

## Overview

This project explores optimal FastText configurations for enhancing [ispravi.me](https://ispravi.me), a Croatian spellchecker, with semantic intelligence. By training word embeddings on a large Croatian corpus, we enable the spellchecker to suggest contextually appropriate corrections based on semantic similarity.

**Key Focus Areas:**
- ije/je alternation errors (e.g., *riješenje* vs *rješenje*)
- Diacritic confusion (č/ć, š/ž)
- General typos and misspellings
- Morphological variations (Croatian has 7 grammatical cases)

## Team

| Name | Role |
|------|------|
| Nikša Brala | Team Member |
| Teo Matošević | Team Member |
| Vitomir Brebrić | Team Member |

**Institution:** Faculty of Electrical Engineering and Computing (FER), University of Zagreb

## Project Structure

```
diplomski_projekt/
├── data/
│   ├── corpus_clean.txt          # Cleaned Croatian corpus (2.5GB, ~400M words)
│   └── eval/                     # Evaluation datasets
│       ├── ije_je_errors.json
│       ├── diacritic_errors.json
│       ├── general_typos.json
│       ├── similarity_pairs.json
│       └── morphological.json
├── scripts/
│   ├── train_optimized.py        # Main training script
│   ├── evaluate_model.py         # Model evaluation
│   ├── compare_models.py         # Model comparison
│   ├── download_baseline.sh      # Download Facebook baseline
│   ├── corpus_cleaner.py         # Text preprocessing
│   └── build_corpus.py           # Corpus extraction
├── models/                       # Trained models (not in git)
├── results/                      # Training & evaluation results
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.8+
- ~32GB RAM for training (models use ~7GB peak)
- ~50GB disk space for models

### Installation

```bash
# Clone repository
git clone <repository-url>
cd diplomski_projekt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Corpus

The corpus (`data/corpus_clean.txt`) is not included in the repository due to size. Contact the team for access or build your own using:

```bash
python scripts/build_corpus.py --input <raw_data> --output data/corpus_clean.txt
```

## Usage

### Check Training Status

```bash
python scripts/train_optimized.py --status
```

Shows all model configurations and their completion status.

### Train Models

```bash
# Train a specific model by name
python scripts/train_optimized.py --name sg_d200_ws5_e3_mc2

# Train by tier
python scripts/train_optimized.py --tier 0          # Quick models (~2h each)
python scripts/train_optimized.py --tier 1          # Standard models (~11h each)
python scripts/train_optimized.py --tier 2          # Experimental models

# Train with limit (useful for incremental training)
python scripts/train_optimized.py --tier 0 --limit 2  # Train only 2 models

# Train all models
python scripts/train_optimized.py --all
```

### Evaluate Models

```bash
# Evaluate all trained models
python scripts/evaluate_model.py --all-models

# Evaluate specific model
python scripts/evaluate_model.py --model models/ft_sg_d200_ws5_e3_mc2.bin

# Compare with Facebook baseline
bash scripts/download_baseline.sh
python scripts/evaluate_model.py --baseline
```

### Compare Models

```bash
python scripts/compare_models.py
```

## Model Configurations

### Tier 0: Quick (~2 hours each)
Fast iteration with reduced dimensions and epochs.

| Model Name | dim | ws | epoch | Notes |
|------------|-----|-----|-------|-------|
| sg_d200_ws5_e3_mc2 | 200 | 5 | 3 | Balanced quick |
| sg_d100_ws5_e3_mc2 | 100 | 5 | 3 | Smallest, fastest |
| sg_d200_ws10_e3_mc2 | 200 | 10 | 3 | Larger window |
| cbow_d200_ws5_e3_mc2 | 200 | 5 | 3 | CBOW variant |

### Tier 1: Standard (~11 hours each)
Full training based on Seminar 2 findings.

| Model Name | dim | ws | epoch | Notes |
|------------|-----|-----|-------|-------|
| sg_d300_ws10_e10_mc2 | 300 | 10 | 10 | Best from Seminar 2 |
| sg_d300_ws10_e15_mc2 | 300 | 10 | 15 | More epochs |
| sg_d300_ws15_e10_mc2 | 300 | 15 | 10 | Larger window |
| sg_d300_ws10_e10_mc2_minn2_maxn7 | 300 | 10 | 10 | Wider n-grams |
| cbow_d300_ws10_e10_mc2 | 300 | 10 | 10 | CBOW architecture |
| sg_d300_ws10_e10_mc2_lr01 | 300 | 10 | 10 | Lower learning rate |

### Tier 2: Experimental (~11 hours each)
Learning rate and negative sampling variations.

| Model Name | dim | ws | epoch | Notes |
|------------|-----|-----|-------|-------|
| sg_d300_ws10_e10_mc2_lr025 | 300 | 10 | 10 | Higher LR (0.25) |
| sg_d300_ws10_e10_mc2_neg10 | 300 | 10 | 10 | 10 negative samples |
| sg_d300_ws10_e10_mc2_neg15 | 300 | 10 | 10 | 15 negative samples |
| sg_d200_ws10_e15_mc2 | 200 | 10 | 15 | Smaller dim, more epochs |

### Common Parameters
- **minCount**: 2 (include rare words)
- **minn**: 3 (minimum n-gram length)
- **maxn**: 6 (maximum n-gram length)

## Parallel Training

For distributed training across multiple machines:

1. Check available models: `python scripts/train_optimized.py --status`
2. Coordinate with team (e.g., "I'll train sg_d300_ws10_e5_mc2")
3. Train specific model: `python scripts/train_optimized.py --name <model_name>`
4. Upload `.bin` file to shared storage (Google Drive)
5. Download teammates' models to `models/` folder

## Evaluation Metrics

Models are evaluated on:

| Metric | Weight | Description |
|--------|--------|-------------|
| ije/je MRR | 20% | Mean Reciprocal Rank for ije/je errors |
| Diacritic MRR | 20% | MRR for č/ć/š/ž confusion |
| General Typo MRR | 20% | MRR for common typos |
| Semantic Similarity | 20% | Correlation with human judgments |
| Morphological Coverage | 10% | Recognition of word forms |
| Model Size & Speed | 10% | Practical deployment considerations |

## Croatian Language Considerations

### Why FastText for Croatian?

1. **Subword Information**: Character n-grams capture morphological patterns
2. **OOV Handling**: Can generate vectors for unseen words
3. **Diacritics**: N-grams help distinguish č/ć, š/ž (though not perfectly)

### Known Limitations

- **č/ć Confusion**: FastText sees these as similar due to shared n-grams
- **Single-character errors**: May need post-processing rules
- **Context-free**: Embeddings don't capture sentence context

### Parameter Choices

- **dim=300**: Higher dimensions capture more semantic nuance
- **ws=10-15**: Larger windows for morphologically rich languages
- **minCount=2**: Include rare words (important for proper nouns)
- **minn=2, maxn=6-7**: Capture Croatian suffixes and prefixes

## Results

*Results will be added after model evaluation is complete.*

## References

- [FastText](https://fasttext.cc/) - Library for text classification and representation
- [ispravi.me](https://ispravi.me) - Croatian spellchecker
- Bojanowski, P., et al. (2017). "Enriching Word Vectors with Subword Information"

## License

This project is part of academic research at FER, University of Zagreb.
