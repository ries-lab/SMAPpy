"""The padding drawn into training frames (`decode_smlm.neuralfitter.edge_pad`)."""
import pytest

torch = pytest.importorskip("torch")

from decode_smlm.generic.emitter import EmitterSet  # noqa: E402
from decode_smlm.neuralfitter.edge_pad import EdgePad  # noqa: E402
from decode_smlm.utils.types import RecursiveNamespace  # noqa: E402


def test_the_bands_get_the_value_and_lose_the_emitters_under_them():
    pad = EdgePad(p=1.0, max_px=7, value=-1.0)
    pad.widths = lambda: [2, 0, 0, 3]                  # 2 rows at the top, 3 columns at the right
    frames = torch.zeros((1, 40, 40))
    em = EmitterSet(xyz=torch.tensor([[1.0, 20.0, 0.0], [1.6, 20.0, 0.0], [20.0, 36.4, 0.0], [20.0, 37.6, 0.0]]),
                    phot=torch.ones(4), frame_ix=torch.zeros(4, dtype=torch.long), xy_unit="px")
    out, kept = pad.forward(frames, em)
    assert (out[0, :2] == -1).all() and (out[0, :, 37:] == -1).all()
    assert (out[0, 2:, :37] == 0).all()
    assert (frames == 0).all()                         # the input untouched
    assert kept.xyz[:, :2].tolist() == [[1.600000023841858, 20.0], [20.0, 36.400001525878906]]


def test_no_band_leaves_frames_and_emitters_as_they_are():
    pad = EdgePad(p=0.0, max_px=7, value=-1.0)
    frames = torch.rand((1, 16, 16))
    out, _ = pad.forward(frames, None)
    assert out is frames


def test_it_is_off_unless_the_job_asks_and_reads_the_job_s_settings():
    assert EdgePad.parse(RecursiveNamespace(HyperParameter={"edge_pad": None})) is None
    pad = EdgePad.parse(RecursiveNamespace(HyperParameter={"edge_pad": {"p": 0.3, "max_px": 5, "value": -2.0}}))
    assert pad.as_dict() == {"p": 0.3, "max_px": 5, "value": -2.0}
    widths = [w for _ in range(2000) for w in pad.widths()]
    assert max(widths) == 5 and 0.25 < sum(w > 0 for w in widths) / len(widths) < 0.35
