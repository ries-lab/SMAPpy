"""``smappy-gui [FILE.hdf5]``: the Qt GUI.  Needs ``pip install smappy-smlm[gui]``.

``smappy-gui --handoff FILE.json`` is how *Open in a new window* starts it,
with the session another instance handed over (`smappy.handoff`).
"""


def main() -> None:
    from ..gui.app import main as run
    raise SystemExit(run())


if __name__ == "__main__":
    main()
