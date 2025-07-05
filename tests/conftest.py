"""global fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """Resource fixture."""
    return Path(__file__).parent / "resources"


@pytest.fixture
def returns(resource_dir) -> pd.DataFrame:
    """Fixture that returns a DataFrame with Meta returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pd.DataFrame: A DataFrame containing Date and Meta returns.

    """
    # Only feed in frames. No series.
    dframe = pd.read_csv(resource_dir / "meta.csv", parse_dates=["Date"])
    return dframe[["Date", "Meta"]]


@pytest.fixture
def benchmark(resource_dir) -> pd.DataFrame:
    """Fixture that returns a DataFrame with benchmark returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pd.DataFrame: A DataFrame containing Date and SPY benchmark returns.

    """
    dframe = pd.read_csv(resource_dir / "benchmark.csv", parse_dates=["Date"])
    return dframe[["Date", "SPY -- Benchmark"]]



@pytest.fixture
def portfolio(resource_dir) -> pd.DataFrame:
    """Fixture that returns a DataFrame with portfolio returns.

    Args:
        resource_dir: The resource_dir fixture containing the path to test resources.

    Returns:
        pd.DataFrame: A DataFrame containing Date, AAPL, and META returns.

    """
    # Read the CSV file
    dframe = pd.read_csv(resource_dir / "portfolio.csv", parse_dates=["Date"])
    # Convert columns to appropriate types
    dframe["AAPL"] = pd.to_numeric(dframe["AAPL"], errors="coerce")
    dframe["META"] = pd.to_numeric(dframe["META"], errors="coerce")
    return dframe


@pytest.fixture()
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns
    -------
    Path
        Path to the README.md file

    Raises
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")
