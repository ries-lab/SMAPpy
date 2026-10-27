"""``smappy-gui [FILE.hdf5]``: the Qt GUI.  Needs ``pip install smappy-smlm[gui]``.

``smappy-gui --handoff FILE.json`` is how *Open in a new window* starts it,
with the session another instance handed over (`smappy.handoff`).

``smappy-gui --desktop-icon`` makes an icon on the desktop that starts it, with
SMAPpy's picture, and exits without opening a window (`smappy.shortcut`).
"""
import sys


def main() -> None:
    if "--desktop-icon" in sys.argv[1:]:
        from ..shortcut import make
        print(f"made {make()}")
        raise SystemExit(0)
    from ..gui.app import main as run
    raise SystemExit(run())


if __name__ == "__main__":
    main()
