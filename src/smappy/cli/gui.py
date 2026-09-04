"""``smappy-gui [FILE.hdf5]``: the Qt GUI.  Needs ``pip install smappy-smlm[gui]``."""


def main() -> None:
    from ..gui.app import main as run
    raise SystemExit(run())
