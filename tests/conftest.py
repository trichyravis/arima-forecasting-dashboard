"""
Pytest Configuration - tests/conftest.py

Global pytest fixtures and configuration for all tests.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def sample_prices():
    """Create sample price data for all tests."""
    dates = pd.date_range('2023-01-01', periods=250)
    prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
    return pd.Series(prices, index=dates, name='price')


@pytest.fixture(scope="session")
def sample_returns():
    """Create sample returns data."""
    dates = pd.date_range('2023-01-01', periods=250)
    returns = np.random.randn(250) * 0.02
    return pd.Series(returns, index=dates, name='returns')


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cache").mkdir()
    (data_dir / "results").mkdir()
    return data_dir


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add marker for tests
        if "arima" in item.nodeid:
            item.add_marker(pytest.mark.slow)

