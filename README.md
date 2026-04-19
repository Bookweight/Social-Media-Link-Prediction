# Social Media HW1 -- Directed Graph Link Prediction

Predict hidden follow relationships in a directed social network.

## Project Structure

```
.
├── predict.py          # Entry point -- runs all strategies
├── graph_utils.py      # Graph construction, negative sampling, precomputation
├── features.py         # 25+ graph feature engineering
├── models.py           # 4 model strategies (Heuristic, LightGBM, Node2Vec, Ensemble)
├── pyproject.toml      # Project config & dependencies
├── train.csv           # Training data (known edges)
├── test.csv            # Test pairs to predict
├── sample_submission.csv
└── output/             # Generated submission CSVs (gitignored)
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run all 4 strategies
uv run predict.py
```

## Model Strategies

| # | Strategy | Method | CV AUC |
|---|----------|--------|--------|
| 1 | Heuristic | Adamic-Adar (normalized) | N/A |
| 2 | LightGBM | 25+ graph features, 5-fold CV | ~0.993 |
| 3 | Node2Vec | Embedding + Logistic Regression | ~0.996 |
| 4 | Ensemble | Weighted rank average of 1-3 | -- |

## Output

4 files in `output/`:
- `submission_heuristic.csv`
- `submission_lgbm.csv`
- `submission_node2vec.csv`
- `submission_ensemble.csv`

## Evaluation

ROC AUC (Area Under the Receiver Operating Characteristic Curve)
