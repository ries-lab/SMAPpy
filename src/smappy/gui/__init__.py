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
    if QApplication.instance() is None:
        name_the_mac_app()
    app = QApplication.instance() or QApplication(
        list(argv) if argv is not None else [sys.argv[0]])
    app.setApplicationName("SMAPpy")
    apply_style(app)
    collect_on_gui_thread(app)
    install_icon(app)
    return app


def name_the_mac_app(name: str = "SMAPpy") -> None:
    """The name in bold at the left of a Mac's menu bar, which is the
    bundle's: run from a script, that is the interpreter's ("Python3.12").

    Set in the main bundle's info dictionary before the application starts,
    through the Objective-C runtime with ctypes, so that no PyObjC is needed.
    Anything going wrong leaves the old name, which is all it costs.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def send(receiver, selector, *args):
            # objc_msgSend is called with the exact prototype of each message:
            # on arm64 a variadic call would pass the arguments wrongly
            send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    *([ctypes.c_void_p] * len(args)))(
                ("objc_msgSend", objc))
            return send(receiver, objc.sel_registerName(selector), *args)

        def text(value: str):
            string = objc.objc_getClass(b"NSString")
            make = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_char_p)(("objc_msgSend", objc))
            return make(string, objc.sel_registerName(b"stringWithUTF8String:"),
                        value.encode())

        bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
        info = send(bundle, b"infoDictionary")
        if info:
            send(info, b"setObject:forKey:", text(name), text("CFBundleName"))
    except Exception:
        pass
