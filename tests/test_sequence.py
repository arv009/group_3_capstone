"""Runs the PyTorch sequence tests (tests/sequence_tests.py) in a separate process.

PyTorch and LightGBM ship clashing OpenMP runtimes on macOS: once another test file has imported
LightGBM (via secondlook.modeling), a GRU forward pass in the same process crashes. A child process
that imports only PyTorch avoids it.
"""
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def test_sequence_suite_in_its_own_process():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TESTS / "sequence_tests.py")],
        cwd=TESTS.parent, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
