"""Train reproducible customer segmentation and anomaly-detection artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Artifact generation must work in Docker, CI, and ordinary terminals with no GUI display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

from .artifacts import SegmentationBundle

RAW_COLUMNS = {
    "invoice": ("invoice", "invoiceno", "invoicenumber"),
    "stock_code": ("stockcode", "sku", "productcode"),
    "quantity": ("quantity", "qty"),
    "invoice_date": ("invoicedate", "date"),
    "price": ("price", "unitprice"),
    "customer_id": ("customerid", "custid"),
}
FEATURE_COLUMNS = (
    "tx_count",
    "spend_sum",
    "item_qty_sum",
    "basket_size_mean",
    "recency_days",
)


@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    output_dir: Path
    seed: int = 1337
    k_min: int = 2
    k_max: int = 8
    contamination: float = 0.03
    max_rows: int | None = None


def repository_root(start: Path | None = None) -> Path:
    """Find a checkout root without relying on notebook working-directory behavior."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "online_retail_II.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not find a repository checkout with online_retail_II.csv.")


def _normalise_column(column: str) -> str:
    return "".join(character for character in column.lower() if character.isalnum())


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map known Online Retail column spelling variants to a strict internal schema."""
    lookup: dict[str, list[str]] = {}
    for column in frame.columns:
        lookup.setdefault(_normalise_column(column), []).append(column)
    rename: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for canonical, aliases in RAW_COLUMNS.items():
        matches = [column for alias in aliases for column in lookup.get(alias, [])]
        if not matches:
            missing.append(canonical)
        elif len(matches) > 1:
            ambiguous[canonical] = matches
        else:
            rename[matches[0]] = canonical
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(frame.columns)}")
    if ambiguous:
        details = ", ".join(f"{field}={columns}" for field, columns in ambiguous.items())
        raise ValueError(f"Ambiguous aliases for required columns: {details}")
    return frame.rename(columns=rename)


def load_transactions(data_path: Path, max_rows: int | None, seed: int) -> pd.DataFrame:
    """Load and clean sale transactions, excluding cancellations and invalid rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    raw = pd.read_csv(data_path, encoding="latin-1", low_memory=False)
    frame = canonicalize_columns(raw)
    frame["invoice"] = frame["invoice"].astype("string").str.strip()
    frame["stock_code"] = frame["stock_code"].astype("string").str.strip()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    customer_id = pd.to_numeric(frame["customer_id"], errors="coerce")
    valid_customer_id = (
        customer_id.notna() & (customer_id > 0) & np.isclose(customer_id.fillna(0) % 1, 0)
    )
    frame = frame.loc[valid_customer_id].copy()
    frame["customer_id"] = customer_id.loc[valid_customer_id].astype("int64")
    frame = frame.loc[frame["invoice"].notna() & frame["invoice"].ne("")].copy()
    frame = frame.loc[frame["stock_code"].notna() & frame["stock_code"].ne("")].copy()
    frame = frame.loc[~frame["invoice"].str.upper().str.startswith("C", na=False)].copy()
    frame = frame.dropna(subset=["invoice_date", "quantity", "price"])
    frame = frame.loc[(frame["quantity"] > 0) & (frame["price"] > 0)].copy()
    if frame.empty:
        raise ValueError("No valid non-cancelled sale transactions remain after cleaning.")
    frame["line_total"] = frame["quantity"] * frame["price"]
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive when supplied.")
        if max_rows < len(frame):
            # Preserve whole baskets so a fast run does not distort basket-size features.
            invoice_sizes = frame.groupby("invoice", observed=True, sort=False).size()
            shuffled_invoices = invoice_sizes.sample(frac=1, random_state=seed)
            selected_invoices: list[str] = []
            selected_rows = 0
            for invoice, invoice_rows in shuffled_invoices.items():
                if selected_rows + invoice_rows <= max_rows:
                    selected_invoices.append(invoice)
                    selected_rows += invoice_rows
                if selected_rows == max_rows:
                    break
            if not selected_invoices:
                raise ValueError(
                    "max_rows is smaller than every complete invoice in the cleaned dataset."
                )
            frame = frame.loc[frame["invoice"].isin(selected_invoices)].copy()
    return frame.loc[
        :,
        ["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id", "line_total"],
    ]


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create non-negative, customer-level features for clustering and anomaly scoring."""
    required = {"invoice", "stock_code", "quantity", "invoice_date", "customer_id", "line_total"}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Transactions missing required cleaned columns: {missing}")
    customer = (
        transactions.groupby("customer_id", observed=True)
        .agg(
            tx_count=("invoice", "nunique"),
            spend_sum=("line_total", "sum"),
            item_qty_sum=("quantity", "sum"),
            last_date=("invoice_date", "max"),
        )
        .reset_index()
    )
    basket_sizes = (
        transactions.groupby(["customer_id", "invoice"], observed=True)
        .size()
        .rename("items_per_invoice")
        .reset_index()
    )
    basket_mean = (
        basket_sizes.groupby("customer_id", observed=True)["items_per_invoice"]
        .mean()
        .rename("basket_size_mean")
        .reset_index()
    )
    customer = customer.merge(basket_mean, on="customer_id", how="left")
    analysis_date = transactions["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    customer["recency_days"] = (
        analysis_date - customer.pop("last_date")
    ).dt.total_seconds() / 86_400
    customer["basket_size_mean"] = customer["basket_size_mean"].fillna(0.0)
    feature_columns = list(FEATURE_COLUMNS)
    customer.loc[:, feature_columns] = customer.loc[:, feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    customer = customer.dropna(subset=feature_columns).reset_index(drop=True)
    if customer.empty or (customer.loc[:, feature_columns] < 0).any().any():
        raise ValueError("Customer features must be present and non-negative.")
    # Low recency is good; this score is descriptive only and intentionally excluded from clustering.
    customer["rfm_score"] = (
        customer["recency_days"].rank(ascending=False, pct=True) * 0.34
        + customer["tx_count"].rank(pct=True) * 0.33
        + customer["spend_sum"].rank(pct=True) * 0.33
    )
    return customer


def transformed_feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, RobustScaler]:
    """Log-transform long-tailed measures and scale robustly before distance-based clustering."""
    raw_matrix = features.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(raw_matrix).all() or (raw_matrix < 0).any():
        raise ValueError("Features must be finite and non-negative.")
    scaler = RobustScaler()
    return scaler.fit_transform(np.log1p(raw_matrix)), scaler


def sampled_silhouette_score(matrix: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Compute a bounded, deterministic silhouette score without dropping a rare cluster."""
    max_samples = 2_000
    if len(matrix) <= max_samples:
        return float(silhouette_score(matrix, labels))

    generator = np.random.default_rng(seed)
    label_values = np.unique(labels)
    representatives = np.array(
        [generator.choice(np.flatnonzero(labels == label)) for label in label_values], dtype=int
    )
    available_indices = np.setdiff1d(np.arange(len(matrix)), representatives, assume_unique=False)
    remaining_size = max_samples - len(representatives)
    sampled_indices = np.concatenate(
        [representatives, generator.choice(available_indices, size=remaining_size, replace=False)]
    )
    return float(silhouette_score(matrix[sampled_indices], labels[sampled_indices]))


def cluster_metrics(matrix: np.ndarray, config: RunConfig) -> tuple[KMeans, pd.DataFrame]:
    """Select a K-Means solution by silhouette score, retaining complementary metrics."""
    distinct_rows = len(np.unique(matrix, axis=0))
    upper_k = min(config.k_max, len(matrix) - 1, distinct_rows)
    if config.k_min < 2 or config.k_min > upper_k:
        raise ValueError(
            "Need enough customers and distinct feature rows for the requested cluster range "
            f"(at least {config.k_min} distinct rows and {config.k_min + 1} customers)."
        )
    rows: list[dict[str, float | int]] = []
    fitted: dict[int, KMeans] = {}
    for k in range(config.k_min, upper_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=config.seed)
        labels = model.fit_predict(matrix)
        fitted[k] = model
        rows.append(
            {
                "k": k,
                "silhouette": sampled_silhouette_score(matrix, labels, config.seed),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    best_k = int(metrics.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    return fitted[best_k], metrics


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_figures(
    metrics: pd.DataFrame, profiles: pd.DataFrame, anomaly_scores: pd.DataFrame, output_dir: Path
) -> None:
    """Write separate diagnostic figures without mixing incompatible metric scales."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(metrics["k"], metrics["silhouette"], marker="o")
    axes[0].set(title="Silhouette by k", xlabel="k", ylabel="Silhouette")
    axes[1].plot(metrics["k"], metrics["davies_bouldin"], marker="o")
    axes[1].set(title="Davies–Bouldin by k", xlabel="k", ylabel="Score (lower is better)")
    axes[2].plot(metrics["k"], metrics["calinski_harabasz"], marker="o")
    axes[2].set(title="Calinski–Harabasz by k", xlabel="k", ylabel="Score")
    fig.tight_layout()
    fig.savefig(output_dir / "clustering_metrics.png", dpi=200)
    plt.close(fig)

    profile_values = profiles.loc[:, list(FEATURE_COLUMNS)]
    standardized = (profile_values - profile_values.mean()) / profile_values.std(ddof=0).replace(
        0, 1
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    image = ax.imshow(standardized.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=35, ha="right")
    ax.set_yticks(range(len(profiles)), [f"Cluster {value}" for value in profiles["cluster"]])
    ax.set_title("Cluster profiles (standardized means)")
    fig.colorbar(image, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_profiles.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(anomaly_scores["anomaly_score"], bins=40)
    ax.set(
        title="Isolation Forest anomaly-score distribution",
        xlabel="Higher = more unusual",
        ylabel="Customers",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_score_distribution.png", dpi=200)
    plt.close(fig)


def run(config: RunConfig) -> dict[str, Any]:
    """Build all exported artifacts from an input transaction CSV."""
    try:
        root = repository_root(config.data_path.parent)
    except FileNotFoundError:
        root = config.data_path.parent
    transactions = load_transactions(config.data_path, config.max_rows, config.seed)
    customer_features = build_customer_features(transactions)
    matrix, scaler = transformed_feature_matrix(customer_features)
    kmeans, metrics = cluster_metrics(matrix, config)
    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination=config.contamination,
        random_state=config.seed,
        n_jobs=-1,
    ).fit(matrix)
    bundle = SegmentationBundle(
        scaler=scaler,
        kmeans=kmeans,
        anomaly_model=anomaly_model,
        feature_columns=FEATURE_COLUMNS,
    )
    assignments = bundle.assign(customer_features.loc[:, list(FEATURE_COLUMNS)])
    customer_features = customer_features.join(assignments)
    customer_features["anomaly_percentile"] = customer_features["anomaly_score"].rank(pct=True)
    labels = customer_features.loc[:, ["customer_id", "cluster"]].copy()
    profiles = (
        customer_features.groupby("cluster", observed=True)[list(FEATURE_COLUMNS)]
        .mean()
        .reset_index()
    )
    profiles.insert(
        1, "customer_count", customer_features.groupby("cluster", observed=True).size().to_numpy()
    )
    anomaly_columns = [
        "customer_id",
        "cluster",
        *FEATURE_COLUMNS,
        "anomaly_score",
        "anomaly_percentile",
        "is_anomaly",
    ]
    anomaly_scores = customer_features.loc[:, anomaly_columns].sort_values(
        "anomaly_score", ascending=False
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(config.output_dir / "customer_features.csv", index=False)
    labels.to_csv(config.output_dir / "cluster_labels.csv", index=False)
    profiles.to_csv(config.output_dir / "cluster_profiles.csv", index=False)
    metrics.to_csv(config.output_dir / "clustering_metrics.csv", index=False)
    anomaly_scores.to_csv(config.output_dir / "anomaly_scores.csv", index=False)
    anomaly_scores.head(10).to_csv(config.output_dir / "anomaly_top10.csv", index=False)
    joblib.dump(bundle, config.output_dir / "segmentation_model.joblib")
    save_figures(metrics, profiles, anomaly_scores, config.output_dir)

    result = {
        "config": {
            **asdict(config),
            "data_path": str(config.data_path),
            "output_dir": str(config.output_dir),
        },
        "dataset_sha256": dataset_sha256(config.data_path),
        "git_revision": git_revision(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "rows": {"transactions": len(transactions), "customers": len(customer_features)},
        "selected_k": int(kmeans.n_clusters),
        "feature_columns": list(FEATURE_COLUMNS),
        "anomaly_count": int(customer_features["is_anomaly"].sum()),
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, help="Transaction CSV; defaults to online_retail_II.csv in a checkout."
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for generated artifacts.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument(
        "--max-rows", type=int, help="Deterministic row sample for faster exploratory runs."
    )
    parser.add_argument("--fast", action="store_true", help="Sample 50,000 rows and search k=2..5.")
    args = parser.parse_args(argv)
    if args.k_min < 2 or args.k_max < args.k_min:
        parser.error("Require 2 <= --k-min <= --k-max.")
    if not 0 < args.contamination < 0.5:
        parser.error("--contamination must be between 0 and 0.5.")
    root: Path | None = None
    if args.data is None:
        try:
            root = repository_root()
        except FileNotFoundError:
            parser.error("--data is required when running outside a repository checkout.")
        data_path = root / "online_retail_II.csv"
    else:
        data_path = args.data
    output_dir = args.output_dir or ((root / "artifacts") if root else (Path.cwd() / "artifacts"))
    k_max = min(args.k_max, 5) if args.fast else args.k_max
    if k_max < args.k_min:
        parser.error("--fast supports --k-min values no higher than 5.")
    return RunConfig(
        data_path=data_path.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        seed=args.seed,
        k_min=args.k_min,
        k_max=k_max,
        contamination=args.contamination,
        max_rows=50_000 if args.fast and args.max_rows is None else args.max_rows,
    )


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps({key: result[key] for key in ("selected_k", "rows", "anomaly_count")}, indent=2)
    )


if __name__ == "__main__":
    main()
