import pandas as pd
import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI
EPOCHS = "1"


@pytest.fixture
def classification_df(data_dir):
    return pd.read_csv(data_dir / "classification" / "mol.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_classification_mol.pt")


@pytest.mark.parametrize("suffix", [".parquet", ".feather"])
def test_train_accepts_table_formats(monkeypatch, classification_df, tmp_path, suffix):
    data_path = tmp_path / f"mol{suffix}"
    match suffix:
        case ".parquet":
            classification_df.to_parquet(data_path, index=False)
        case ".feather":
            classification_df.to_feather(data_path)

    args = [
        "chemprop",
        "train",
        "-i",
        str(data_path),
        "--epochs",
        EPOCHS,
        "--warmup-epochs",
        "0",
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
        "--task-type",
        "classification",
        "--save-dir",
        str(tmp_path / f"train_{suffix.removeprefix('.')}"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("suffix", [".parquet", ".feather"])
def test_predict_outputs_table_formats(
    monkeypatch, classification_df, model_path, tmp_path, suffix
):
    test_path = tmp_path / f"mol{suffix}"
    output_path = tmp_path / f"preds{suffix}"

    match suffix:
        case ".parquet":
            classification_df.to_parquet(test_path, index=False)
        case ".feather":
            classification_df.to_feather(test_path)

    args = [
        "chemprop",
        "predict",
        "-i",
        str(test_path),
        "--model-path",
        model_path,
        "--accelerator",
        "cpu",
        "--output",
        str(output_path),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    readers = {".parquet": pd.read_parquet, ".feather": pd.read_feather}
    df = readers[suffix](output_path)

    assert len(df) == len(classification_df)
    assert df.shape[1] >= classification_df.shape[1]
    individual_path = output_path.with_name(output_path.stem + "_individual" + suffix)
    assert not individual_path.exists()
