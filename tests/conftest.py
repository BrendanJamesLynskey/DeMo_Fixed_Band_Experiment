import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment"))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path, monkeypatch):
    """Keep checkpoints written by test runs out of the repository."""
    import train
    monkeypatch.setattr(train, "ROOT", tmp_path)
