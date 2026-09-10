import joblib
import numpy as np
import pandas as pd
import pytest

from retail_segmentation.artifacts import SegmentationBundle
from retail_segmentation.train import (
    FEATURE_COLUMNS,
    RunConfig,
    build_customer_features,
    canonicalize_columns,
    cluster_metrics,
    load_transactions,
    parse_args,
    run,
)


def make_transactions() -> pd.DataFrame:
    rows = []
    for customer_id in range(100, 110):
        for invoice_number in range(2):
            rows.append(
                {
                    "Invoice": f"{customer_id}{invoice_number}",
                    "StockCode": f"SKU{invoice_number}",
                    "Quantity": customer_id - 95 + invoice_number,
                    "InvoiceDate": f"2010-01-{invoice_number + 1:02d} 10:00:00",
                    "Price": 2.5 + invoice_number,
                    "Customer ID": float(customer_id),
                }
            )
    rows.extend(
        [
            {
                "Invoice": "C1000",
                "StockCode": "RETURN",
                "Quantity": -1,
                "InvoiceDate": "2010-01-03 10:00:00",
                "Price": 2.5,
                "Customer ID": 100.0,
            },
            {
                "Invoice": "bad",
                "StockCode": "BAD",
                "Quantity": 0,
                "InvoiceDate": "2010-01-03 10:00:00",
                "Price": 2.5,
                "Customer ID": 101.0,
            },
            {
                "Invoice": None,
                "StockCode": "MISSING-INVOICE",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 10:00:00",
                "Price": 2.5,
                "Customer ID": 102.0,
            },
            {
                "Invoice": "bad-customer",
                "StockCode": "BAD-CUSTOMER",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 10:00:00",
                "Price": 2.5,
                "Customer ID": 0.0,
            },
        ]
    )
    return pd.DataFrame(rows)


def test_canonicalize_columns_requires_schema():
    with pytest.raises(ValueError, match="Missing required columns"):
        canonicalize_columns(pd.DataFrame({"Invoice": ["1"]}))


def test_canonicalize_columns_rejects_ambiguous_aliases():
    frame = make_transactions().head(1).assign(InvoiceNo="duplicate-invoice-field")

    with pytest.raises(ValueError, match="Ambiguous aliases"):
        canonicalize_columns(frame)


def test_cleaning_and_feature_engineering_exclude_returns(tmp_path):
    path = tmp_path / "retail.csv"
    make_transactions().to_csv(path, index=False)
    cleaned = load_transactions(path, max_rows=None, seed=1337)
    features = build_customer_features(cleaned)

    assert len(cleaned) == 20
    assert len(features) == 10
    assert (features.loc[:, list(FEATURE_COLUMNS)] >= 0).all().all()
    assert "rfm_score" in features


def test_end_to_end_run_writes_reloadable_bundle(tmp_path):
    data_path = tmp_path / "retail.csv"
    output_dir = tmp_path / "artifacts"
    make_transactions().to_csv(data_path, index=False)
    result = run(RunConfig(data_path=data_path, output_dir=output_dir, k_max=3, contamination=0.2))

    assert result["selected_k"] in {2, 3}
    assert (output_dir / "run_metadata.json").exists()
    bundle = joblib.load(output_dir / "segmentation_model.joblib")
    assert isinstance(bundle, SegmentationBundle)
    features = pd.read_csv(output_dir / "customer_features.csv")
    assignments = bundle.assign(features.loc[:, list(FEATURE_COLUMNS)])
    assert len(assignments) == len(features)
    np.testing.assert_array_equal(assignments["cluster"].to_numpy(), features["cluster"].to_numpy())
    np.testing.assert_allclose(
        assignments["anomaly_score"].to_numpy(), features["anomaly_score"].to_numpy()
    )


def test_model_artifact_rejects_an_unexpected_feature(tmp_path):
    data_path = tmp_path / "retail.csv"
    output_dir = tmp_path / "artifacts"
    make_transactions().to_csv(data_path, index=False)
    run(RunConfig(data_path=data_path, output_dir=output_dir, k_max=2, contamination=0.2))
    bundle = joblib.load(output_dir / "segmentation_model.joblib")
    features = pd.read_csv(output_dir / "customer_features.csv")
    invalid_features = features.loc[:, list(FEATURE_COLUMNS)].assign(unexpected=1)

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        bundle.assign(invalid_features)


def test_cluster_selection_rejects_degenerate_customer_features(tmp_path):
    matrix = np.zeros((4, len(FEATURE_COLUMNS)))
    config = RunConfig(data_path=tmp_path / "retail.csv", output_dir=tmp_path / "artifacts")

    with pytest.raises(ValueError, match="distinct feature rows"):
        cluster_metrics(matrix, config)


def test_cluster_selection_keeps_a_rare_cluster_in_the_silhouette_sample(tmp_path):
    matrix = np.zeros((3_001, len(FEATURE_COLUMNS)))
    matrix[3, 0] = 1
    config = RunConfig(
        data_path=tmp_path / "retail.csv", output_dir=tmp_path / "artifacts", k_max=2
    )

    model, metrics = cluster_metrics(matrix, config)

    assert model.n_clusters == 2
    assert len(metrics) == 1


def test_fast_sampling_preserves_complete_invoices(tmp_path):
    rows = [
        {
            "Invoice": invoice,
            "StockCode": f"{invoice}-{line}",
            "Quantity": 1,
            "InvoiceDate": "2010-01-01 10:00:00",
            "Price": 1.0,
            "Customer ID": customer_id,
        }
        for invoice, customer_id in (("A", 1), ("B", 2))
        for line in range(2)
    ]
    data_path = tmp_path / "retail.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    sampled = load_transactions(data_path, max_rows=3, seed=1337)

    assert len(sampled) == 2
    assert sampled.groupby("invoice", observed=True).size().eq(2).all()


def test_cli_accepts_explicit_data_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = parse_args(["--data", "retail.csv"])
    assert config.data_path == (tmp_path / "retail.csv").resolve()
    assert config.output_dir == (tmp_path / "artifacts").resolve()
