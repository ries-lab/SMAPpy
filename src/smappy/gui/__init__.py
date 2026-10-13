"""The Qt GUI (PySide6 + pyqtgraph): ``pip install smappy-smlm[gui]``."""
import sys


def make_app(argv=None):
    """The QApplication, made if there is none, set up as every window wants.

    One place for what each entry point used to repeat: the style of the
    editable fields, garbage collection on the GUI thread (`collector`: off
    everywhere else, or a widget can be destroyed on a worker), and the icon.
    The session's identity is not here, because it needs the session:
    an entry point that makes one calls `identity.install_identity` too.
    """
    from PySide6.QtWidgets import QApplication

    from .collector import collect_on_gui_thread
    from .icon import install_icon
    from .widgets import apply_style
    app = QApplication.instance() or QApplication(
        list(argv) if argv is not None else [sys.argv[0]])
    apply_style(app)
    collect_on_gui_thread(app)
    install_icon(app)
    return app
