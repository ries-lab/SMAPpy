"""The DECODE fitter with its network on a training host (`smappy.decode_remote`, `decode_smlm.serve`).

ssh is played by starting ``python -m decode_smlm.cli serve`` here: the worker, the messages and the
fitter's blocks are the real ones, so the table must be the local fit's exactly.
"""
import subprocess
import sys

import numpy as np
import pytest

from smappy.training import jobs

from .test_fit_plugin import _same, fit, tiny  # noqa: F401  (tiny is a fixture)


@pytest.fixture
def host(monkeypatch):
    started = []
    real = subprocess.Popen

    def popen(command, **kw):
        assert command[0] == "ssh" and command[-1] == "decode serve"
        started.append(command)
        return real([sys.executable, "-m", "decode_smlm.cli", "serve"], **kw)
    from smappy import decode_remote
    monkeypatch.setattr(decode_remote.subprocess, "Popen", popen)
    jobs.set_hosts([jobs.Host("box", "box.example", command="decode")])
    return started


def test_a_fit_with_its_network_on_a_host_is_the_fit_here(tiny, host):  # noqa: F811
    here = fit(tiny["tif"], tiny["model"], **{"source.chunk": 7})
    there = fit(tiny["tif"], tiny["model"], **{"source.chunk": 7, "model.where": "box"})
    assert len(host) == 1
    _same(there, here)


def test_the_workers_error_reaches_the_fit(tiny, host, tmp_path):  # noqa: F811
    from smappy.decode_remote import RemoteFitError, RemotePredictor
    p = RemotePredictor(jobs.host("box"), tiny["model"], "cpu", batch_size=1)
    try:
        assert p.window == 1 and p.device == "box: cpu"
        with pytest.raises(ValueError, match="frames"):
            p(np.zeros((2, 32, 32), np.float32))                      # refused here: no frame has context
        with pytest.raises(RemoteFitError, match="box: "):
            p._ask([(b"FRMS", b"not frames")], b"COLS")                # and there, said back
        cols = p(np.ones((5, 32, 32), np.float32) * 10, 4)
        assert set(cols) >= {"frame", "x_pix", "y_pix", "prob"}
    finally:
        p.close()
    assert p.process.poll() is not None


def test_a_live_fit_and_an_unknown_host_are_refused_before_anything_is_sent(tiny, host):  # noqa: F811
    with pytest.raises(ValueError, match="no training host 'nowhere'"):
        fit(tiny["tif"], tiny["model"], **{"model.where": "nowhere"})
    with pytest.raises(ValueError, match="live fit"):
        fit(tiny["tif"], tiny["model"], **{"model.where": "box", "source.live": True})
    assert host == []
