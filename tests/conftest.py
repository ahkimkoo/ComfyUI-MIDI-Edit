# tests/conftest.py
"""Pytest configuration shared across the alignment test-suite.

Registers the ``slow`` marker so ``@pytest.mark.slow`` does not raise
``PytestUnknownMarkWarning`` and can be filtered via ``-m slow`` /
``-m "not slow"`` on the command line.
"""


def pytest_configure(config):
    """Register custom markers during pytest configuration."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
