"""The Train window's preparation figure with torch: frames as the training draws them, beside the data."""
import numpy as np
import pytest
from test_train_window import _prepare_settings, movie  # noqa: F401  (movie is a fixture)


def test_the_simulated_frames_are_as_bright_as_the_estimates_say_and_drawn_beside_the_data(movie):  # noqa: F811
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from smappy.gui import train_window
    path, cal_path = movie
    prepared = train_window.run_prefit(_prepare_settings(path, cal_path))
    est = prepared.estimates
    params = train_window.method_params(prepared, est)
    frames = train_window.simulated_frames(params, str(cal_path), n_frames=8)
    assert frames.shape == (8, 40, 40)
    lo, hi = params["Simulation"]["bg_uniform"]
    assert lo * 0.8 < np.median(frames) < hi * 1.2                 # the background the training draws
    w = train_window.TrainWindow()
    try:
        w.show_prepared(prepared)                                  # starts the window's own simulation
        import time
        deadline = time.monotonic() + 60
        while w.simulated is None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)
        assert w.simulated is not None and w.simulated.shape[1:] == (40, 40)
        w.simulated = frames
        w.draw_prepared()
        assert len(w.prepare_figure.axes) == 6
        assert [a.get_title() for a in w.prepare_figure.axes[3:5]] == ["data", "simulated"]
    finally:
        w.timer.stop()
        w.close()
