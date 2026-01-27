import fasttext
import os

# Put do korpusa
CORPUS_PATH = "jutarnji-corpus-2024-2025-clean.txt"

# Definicija parametara koje ćemo testirati
param_grid = [
    {"model": "skipgram", "dim": 100, "ws": 5, "epoch": 5, "minCount": 5},
    {"model": "skipgram", "dim": 300, "ws": 5, "epoch": 5, "minCount": 5},
    {"model": "cbow",     "dim": 100, "ws": 5, "epoch": 5, "minCount": 5},
    {"model": "cbow",     "dim": 300, "ws": 5, "epoch": 5, "minCount": 5},
    {"model": "skipgram", "dim": 100, "ws": 10, "epoch": 10, "minCount": 3},
    {"model": "skipgram", "dim": 300, "ws": 10, "epoch": 10, "minCount": 3}
]

os.makedirs("models", exist_ok=True)

for i, params in enumerate(param_grid):
    print(f"Treniranje modela {i+1}/{len(param_grid)}: {params}")
    model = fasttext.train_unsupervised(
        input=CORPUS_PATH,
        model=params["model"],
        dim=params["dim"],
        ws=params["ws"],
        epoch=params["epoch"],
        minCount=params["minCount"]
    )
    model.save_model(f"models/ft_model_{i+1}.bin")