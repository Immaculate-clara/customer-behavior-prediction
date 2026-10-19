"""Train the deep learning models: ANN and LSTM churn classifiers and an autoencoder.

* ANN          - feed-forward network on the same scaled features as the classical models.
* LSTM         - recurrent network on each customer's monthly purchase sequence
                 (orders and revenue per 30-day bucket, oldest to newest).
* Autoencoder  - learns to reconstruct normal customers; a high reconstruction
                 error flags unusual behaviour (anomaly detection).

All models use the same train/val/test split as the other tasks. Early stopping
and the decision threshold use the VALIDATION split only; the test split is
used once for the final scores.

Run:  python -m src.models.train_deep [--models ann,lstm,autoencoder]
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.build_features import TREND_BUCKET_DAYS, _cutoff_from_config
from src.models.data_prep import CHURN_TARGET, ID_COLUMN, ModelData, make_preprocessor, prepare_data
from src.models.evaluate import classification_metrics, log_run, select_threshold
from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_PREFIX = "dl_"
MODEL_NAMES = ("ann", "lstm", "autoencoder")


# --------------------------------------------------------------------------- #
# Settings (every value has a default, so missing config keys never break training)
# --------------------------------------------------------------------------- #
def _setting(section: dict, names: tuple[str, ...], default: Any) -> Any:
    """Return the first key from ``names`` found in ``section``, else ``default``."""
    for name in names:
        if name in section and section[name] is not None:
            return section[name]
    return default


def _as_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _training_settings(section: dict) -> dict[str, Any]:
    return {
        "epochs": int(_setting(section, ("epochs", "max_epochs"), 100)),
        "batch_size": int(_setting(section, ("batch_size",), 64)),
        "lr": float(_setting(section, ("learning_rate", "lr"), 1e-3)),
        "weight_decay": float(_setting(section, ("weight_decay",), 1e-4)),
        "patience": int(_setting(section, ("patience", "early_stopping_patience"), 10)),
    }


def _section(name: str) -> dict:
    return get_config_value(f"deep_learning.{name}", {}) or {}


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class ChurnANN(nn.Module):
    """Fully connected network that outputs one churn logit per customer."""

    def __init__(self, n_features: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for units in hidden_layers:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ChurnLSTM(nn.Module):
    """LSTM over the purchase sequence; the last hidden state feeds a churn logit."""

    def __init__(self, n_channels: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


class Autoencoder(nn.Module):
    """Symmetric encoder/decoder that squeezes the features through a bottleneck."""

    def __init__(self, n_features: int, hidden_layers: list[int], latent_dim: int):
        super().__init__()
        sizes = [n_features, *hidden_layers, latent_dim]
        encoder: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            encoder.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                encoder.append(nn.ReLU())
        reverse = sizes[::-1]
        decoder: list[nn.Module] = []
        for i in range(len(reverse) - 1):
            decoder.append(nn.Linear(reverse[i], reverse[i + 1]))
            if i < len(reverse) - 2:
                decoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Generic training loop with early stopping
# --------------------------------------------------------------------------- #
def _tensor(values, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=dtype)


def fit_network(
    model: nn.Module,
    loss_fn: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train with Adam; keep the weights from the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*train),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])

    best_loss, best_state, best_epoch, waited = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val[0]), val[1]))

        if val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= settings["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_run": epoch}


def _predict_proba(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(inputs)).numpy()


def _score_classifier(model: nn.Module, inputs: dict[str, torch.Tensor], data: ModelData, min_recall: float) -> dict[str, Any]:
    """Pick the threshold on validation, then score validation and test."""
    val_proba = _predict_proba(model, inputs["val"])
    test_proba = _predict_proba(model, inputs["test"])
    threshold = select_threshold(data.y("val"), val_proba, min_recall=min_recall)
    return {
        "threshold": threshold,
        "val": classification_metrics(data.y("val"), val_proba, threshold),
        "test": classification_metrics(data.y("test"), test_proba, threshold),
    }


# --------------------------------------------------------------------------- #
# Purchase sequences for the LSTM
# --------------------------------------------------------------------------- #
def build_sequences(
    transactions: pd.DataFrame,
    customer_ids: pd.Series,
    cutoff: pd.Timestamp,
    n_steps: int,
) -> np.ndarray:
    """Array (customers, n_steps, 2): log orders and signed-log revenue per 30-day bucket.

    Step 0 is the oldest bucket and step ``n_steps - 1`` the most recent one before
    the cutoff. Only rows before the cutoff are used (no leakage).
    """
    history = transactions[transactions["invoice_date"] < cutoff]
    sales = history[~history["is_cancellation"]]
    invoices = sales.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_date=("invoice_date", "min"), revenue=("line_total", "sum")
    )
    bucket = ((cutoff - invoices["invoice_date"]).dt.days // TREND_BUCKET_DAYS).to_numpy()
    keep = bucket < n_steps
    invoices = invoices.loc[keep].assign(step=n_steps - 1 - bucket[keep])

    grouped = invoices.groupby(["customer_id", "step"]).agg(orders=("invoice", "size"), revenue=("revenue", "sum"))
    position = pd.Series(np.arange(len(customer_ids)), index=pd.Index(customer_ids.to_numpy()))
    rows = position.reindex(grouped.index.get_level_values(0)).to_numpy()
    valid = ~np.isnan(rows)

    rows = rows[valid].astype(int)
    steps = grouped.index.get_level_values(1).to_numpy()[valid].astype(int)
    revenue = grouped["revenue"].to_numpy()[valid]

    sequences = np.zeros((len(customer_ids), n_steps, 2), dtype=np.float32)
    sequences[rows, steps, 0] = np.log1p(grouped["orders"].to_numpy()[valid])
    sequences[rows, steps, 1] = np.sign(revenue) * np.log1p(np.abs(revenue))
    return sequences


def _load_transactions() -> pd.DataFrame:
    path = resolve_path(get_config_value("paths.interim_transactions", "data/interim/transactions_clean.parquet"))
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}. Run the cleaner first.")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Individual models
# --------------------------------------------------------------------------- #
def train_ann(data: ModelData, seed: int, min_recall: float, artifacts_dir) -> dict[str, Any]:
    cfg = _section("ann")
    settings = _training_settings(cfg)
    hidden = _as_list(_setting(cfg, ("hidden_layers", "hidden_units", "layers"), [64, 32]))
    dropout = float(_setting(cfg, ("dropout",), 0.3))

    preprocessor = make_preprocessor(scaled=True)
    X = {name: _tensor(preprocessor.fit_transform(data.X(name)) if name == "train" else preprocessor.transform(data.X(name)))
         for name in ("train", "val", "test")}
    y = {name: _tensor(data.y(name).to_numpy()) for name in ("train", "val")}

    torch.manual_seed(seed)
    model = ChurnANN(X["train"].shape[1], hidden, dropout)
    info = fit_network(model, nn.BCEWithLogitsLoss(), (X["train"], y["train"]), (X["val"], y["val"]), settings, seed)
    scores = _score_classifier(model, X, data, min_recall)

    joblib.dump(preprocessor, artifacts_dir / f"{ARTIFACT_PREFIX}ann_preprocessor.joblib")
    torch.save(
        {"state_dict": model.state_dict(), "n_features": X["train"].shape[1], "hidden_layers": hidden, "dropout": dropout},
        artifacts_dir / f"{ARTIFACT_PREFIX}ann.pt",
    )
    return {"scores": scores, "info": info, "params": {"hidden_layers": hidden, "dropout": dropout, **settings}}


def train_lstm(data: ModelData, seed: int, min_recall: float, artifacts_dir) -> dict[str, Any]:
    cfg = _section("lstm")
    settings = _training_settings(cfg)
    n_steps = int(_setting(cfg, ("sequence_length", "timesteps", "n_steps"), get_config_value("features.trend_window_months", 12)))
    hidden_size = _as_list(_setting(cfg, ("hidden_size", "hidden_units", "units"), 32))[0]
    num_layers = int(_setting(cfg, ("num_layers", "layers"), 1))
    dropout = float(_setting(cfg, ("dropout",), 0.2))

    all_ids = data.table[ID_COLUMN]
    sequences = build_sequences(_load_transactions(), all_ids, _cutoff_from_config(), n_steps)
    masks = {name: (data.split == name).to_numpy() for name in ("train", "val", "test")}

    # Standardise each channel with training-set statistics only.
    train_seq = sequences[masks["train"]]
    mean, std = train_seq.mean(axis=(0, 1)), train_seq.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    X = {name: _tensor((sequences[mask] - mean) / std) for name, mask in masks.items()}
    y = {name: _tensor(data.y(name).to_numpy()) for name in ("train", "val")}

    torch.manual_seed(seed)
    model = ChurnLSTM(X["train"].shape[2], hidden_size, num_layers, dropout)
    info = fit_network(model, nn.BCEWithLogitsLoss(), (X["train"], y["train"]), (X["val"], y["val"]), settings, seed)
    scores = _score_classifier(model, X, data, min_recall)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_channels": int(X["train"].shape[2]),
            "n_steps": n_steps,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
        },
        artifacts_dir / f"{ARTIFACT_PREFIX}lstm.pt",
    )
    return {
        "scores": scores,
        "info": info,
        "params": {"n_steps": n_steps, "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, **settings},
    }


def train_autoencoder(data: ModelData, seed: int, artifacts_dir) -> dict[str, Any]:
    cfg = _section("autoencoder")
    settings = _training_settings(cfg)
    hidden = _as_list(_setting(cfg, ("hidden_layers", "hidden_units", "layers"), [16]))
    latent = _as_list(_setting(cfg, ("latent_dim", "encoding_dim", "bottleneck"), 8))[0]
    percentile = float(_setting(cfg, ("anomaly_percentile",), 95))

    preprocessor = make_preprocessor(scaled=True)
    X = {name: _tensor(preprocessor.fit_transform(data.X(name)) if name == "train" else preprocessor.transform(data.X(name)))
         for name in ("train", "val", "test")}

    torch.manual_seed(seed)
    model = Autoencoder(X["train"].shape[1], hidden, latent)
    info = fit_network(model, nn.MSELoss(), (X["train"], X["train"]), (X["val"], X["val"]), settings, seed)

    def errors(name: str) -> np.ndarray:
        with torch.no_grad():
            return ((model(X[name]) - X[name]) ** 2).mean(dim=1).numpy()

    train_err, val_err, test_err = errors("train"), errors("val"), errors("test")
    threshold = float(np.percentile(train_err, percentile))
    churn_test = data.y("test").to_numpy()
    flagged = test_err > threshold

    joblib.dump(preprocessor, artifacts_dir / f"{ARTIFACT_PREFIX}autoencoder_preprocessor.joblib")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": X["train"].shape[1],
            "hidden_layers": hidden,
            "latent_dim": latent,
            "anomaly_threshold": threshold,
        },
        artifacts_dir / f"{ARTIFACT_PREFIX}autoencoder.pt",
    )
    scores = {
        "train_mse": float(train_err.mean()),
        "val_mse": float(val_err.mean()),
        "test_mse": float(test_err.mean()),
        "anomaly_threshold": threshold,
        "test_flagged_share": float(flagged.mean()),
        "test_churn_rate_flagged": float(churn_test[flagged].mean()) if flagged.any() else float("nan"),
        "test_churn_rate_normal": float(churn_test[~flagged].mean()) if (~flagged).any() else float("nan"),
    }
    return {"scores": scores, "info": info, "params": {"hidden_layers": hidden, "latent_dim": latent, **settings}}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(model_names: list[str] | None = None) -> pd.DataFrame:
    """Train the selected deep learning models and save artifacts."""
    names = list(model_names) if model_names else list(MODEL_NAMES)
    unknown = sorted(set(names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}. Available: {list(MODEL_NAMES)}")

    seed = get_seed()
    np.random.seed(seed)
    data = prepare_data()
    min_recall = float(get_config_value("training.targets.recall", 0.70))
    artifacts_dir = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"), create=True)

    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    autoencoder_scores: dict[str, Any] | None = None

    for name in names:
        logger.info("Training %s ...", name)
        if name == "autoencoder":
            result = train_autoencoder(data, seed, artifacts_dir)
            autoencoder_scores = result["scores"]
            log_run(
                {"task": "anomaly_detection", "model": name, **autoencoder_scores, "n_train": len(data.y("train")),
                 "seed": seed, "params": json.dumps(result["params"], default=str)}
            )
            logger.info("autoencoder | epochs %s | val MSE %.4f | test MSE %.4f",
                        result["info"]["epochs_run"], autoencoder_scores["val_mse"], autoencoder_scores["test_mse"])
            continue

        trainer = train_ann if name == "ann" else train_lstm
        result = trainer(data, seed, min_recall, artifacts_dir)
        scores = result["scores"]
        thresholds[name] = scores["threshold"]
        row = {
            "model": name,
            "val_roc_auc": scores["val"]["roc_auc"],
            "test_roc_auc": scores["test"]["roc_auc"],
            "test_recall": scores["test"]["recall"],
            "test_precision": scores["test"]["precision"],
            "test_f1": scores["test"]["f1"],
            "test_accuracy": scores["test"]["accuracy"],
            "threshold": scores["threshold"],
            "best_epoch": result["info"]["best_epoch"],
        }
        rows.append(row)
        log_run(
            {"task": "churn_classification", **row, "n_train": len(data.y("train")), "seed": seed,
             "params": json.dumps(result["params"], default=str)}
        )
        logger.info("%s | best epoch %s | val AUC %.3f | test AUC %.3f | test recall %.3f",
                    name, row["best_epoch"], row["val_roc_auc"], row["test_roc_auc"], row["test_recall"])

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison = comparison.sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
        comparison.to_csv(artifacts_dir / "deep_model_comparison.csv", index=False)

    metadata_path = artifacts_dir / "deep_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"thresholds": {}}
    metadata.update(
        {
            "feature_columns": data.feature_columns,
            "seed": seed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    metadata["thresholds"].update(thresholds)
    if autoencoder_scores is not None:
        metadata["autoencoder"] = autoencoder_scores
    metadata_path.write_text(json.dumps(metadata, indent=2))

    _report(comparison, autoencoder_scores, get_config_value("training.targets", {}))
    return comparison


def _report(comparison: pd.DataFrame, autoencoder: dict[str, Any] | None, targets: dict) -> None:
    if not comparison.empty:
        table = comparison.copy()
        table["meets_auc"] = table["test_roc_auc"] >= float(targets.get("roc_auc", 0.80))
        table["meets_recall"] = table["test_recall"] >= float(targets.get("recall", 0.70))
        print("\nDeep learning churn models (ranked by validation ROC-AUC)")
        print(table.round(3).to_string(index=False))
    if autoencoder is not None:
        print("\nAutoencoder anomaly detection")
        for key, value in autoencoder.items():
            print(f"  {key:<26}{value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep learning models")
    parser.add_argument("--models", default="", help="comma-separated subset, e.g. ann,lstm")
    args = parser.parse_args()
    run([item.strip() for item in args.models.split(",") if item.strip()] or None)