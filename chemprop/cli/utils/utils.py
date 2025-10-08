from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["pop_attr", "read_table", "write_table"]


def pop_attr(o: object, attr: str, *args) -> Any | None:
    """like ``pop()`` but for attribute maps"""
    match len(args):
        case 0:
            return _pop_attr(o, attr)
        case 1:
            return _pop_attr_d(o, attr, args[0])
        case _:
            raise TypeError(f"Expected at most 2 arguments! got: {len(args)}")


def _pop_attr(o: object, attr: str) -> Any:
    val = getattr(o, attr)
    delattr(o, attr)

    return val


def _pop_attr_d(o: object, attr: str, default: Any | None = None) -> Any | None:
    try:
        val = getattr(o, attr)
        delattr(o, attr)
    except AttributeError:
        val = default

    return val


def _to_str(number: float) -> str:
    return f"{number:.6e}"


def format_probability_string(test_preds: np.ndarray) -> np.ndarray:
    axis = test_preds.ndim - 1
    formatted_probability_strings = np.apply_along_axis(
        lambda x: ",".join(map(_to_str, x)), axis, test_preds
    )
    return formatted_probability_strings


def read_table(
    path: PathLike,
    header: int | str | None = "infer",
    index_col: bool | int | None = False,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a table from a file (CSV, Parquet, or Feather).

    Automatically detects file format based on extension and uses the appropriate
    pandas read function.

    Parameters
    ----------
    path : PathLike
        Path to the file to read
    header : int, str, or None, default "infer"
        Row number(s) to use as column names. Use None for no header row.
    index_col : bool, int, or None, default False
        Column(s) to use as the row labels
    nrows : int or None, default None
        Number of rows to read. None reads all rows.

    Returns
    -------
    pd.DataFrame
        The data from the file

    Raises
    ------
    ValueError
        If file format is not supported (must be .csv, .parquet, or .feather)
    ImportError
        If pyarrow is not installed for parquet/feather files
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, header=header, index_col=index_col, nrows=nrows)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                "Reading parquet files requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from e

        # Apply header and index_col logic for parquet
        if header is None:
            df.columns = range(len(df.columns))
        if index_col is not False and index_col is not None:
            df = df.set_index(df.columns[index_col])
        if nrows is not None:
            df = df.head(nrows)
        return df
    elif suffix == ".feather":
        try:
            df = pd.read_feather(path)
        except ImportError as e:
            raise ImportError(
                "Reading feather files requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from e

        # Apply header and index_col logic for feather
        if header is None:
            df.columns = range(len(df.columns))
        if index_col is not False and index_col is not None:
            df = df.set_index(df.columns[index_col])
        if nrows is not None:
            df = df.head(nrows)
        return df
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats are: .csv, .parquet, .feather"
        )


def write_table(
    df: pd.DataFrame,
    path: PathLike,
    index: bool = False,
) -> None:
    """Write a DataFrame to a file (CSV, Parquet, Feather, or Pickle).

    Automatically detects file format based on extension and uses the appropriate
    pandas write function.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to write
    path : PathLike
        Path to the output file
    index : bool, default False
        Whether to write the index

    Raises
    ------
    ValueError
        If file format is not supported (must be .csv, .parquet, .feather, or .pkl)
    ImportError
        If pyarrow is not installed for parquet/feather files
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
        except ImportError as e:
            raise ImportError(
                "Writing parquet files requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from e
    elif suffix == ".feather":
        try:
            # Feather format doesn't support index writing the same way
            if index:
                df = df.reset_index()
            df.to_feather(path)
        except ImportError as e:
            raise ImportError(
                "Writing feather files requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from e
    elif suffix == ".pkl":
        df = df.reset_index(drop=not index)
        df.to_pickle(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats are: .csv, .parquet, .feather, .pkl"
        )
