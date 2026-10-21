"""The Mac's own open panel, files and folders together, several at once.

Qt's file dialog chooses files or one folder, never both: its native mode on
a Mac is an `NSOpenPanel` set up for one or the other.  The panel itself
takes files and folders in one selection, which is what picking a few bead
z-stacks wants -- each a folder of its own, beside single TIFFs.  So the panel
is driven directly, through the Objective-C runtime with ctypes as
`gui.name_the_mac_app` does, and no PyObjC is needed.

Anything going wrong -- not a Mac, a runtime that does not answer, a selector
missing -- returns None, and the caller opens Qt's dialog as before: the
cost of a failure is the old dialog, never no dialog.  Windows' own dialog
cannot mix files and folders either (several folders, or several files), and
Qt's does not offer its several-folders mode; there, and on Linux, the Qt
dialog and dropping on the window are what there is.
"""
from __future__ import annotations

import sys
from typing import List, Optional

#: `NSModalResponseOK`: the panel's *Open*
_OK = 1


def available() -> bool:
    return sys.platform == "darwin"


def choose_files_and_folders(title: str = "", start: str = "") -> Optional[List[str]]:
    """The paths chosen in the Mac's open panel, files and folders, several at once; ``[]`` if cancelled;
    None where the panel cannot be used (not a Mac, or the runtime refused), for the Qt dialog instead.

    Modal, on the GUI thread, as Qt's own native dialog is."""
    if not available():
        return None
    try:
        return _run(title, start)
    except Exception:                    # an unexpected runtime: the Qt dialog, not an error
        return None


def _run(title: str, start: str) -> List[str]:
    import ctypes
    import ctypes.util
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
    objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]

    def message(restype, *argtypes):
        # objc_msgSend with the exact prototype of each message: on arm64 a variadic call, or a BOOL passed
        # as a pointer, passes the arguments wrongly
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", objc))

    def sel(name: bytes):
        return objc.sel_registerName(name)

    send = message(ctypes.c_void_p)
    send_object = message(ctypes.c_void_p, ctypes.c_void_p)
    send_bool = message(None, ctypes.c_bool)
    send_index = message(ctypes.c_void_p, ctypes.c_ulong)
    count_of = message(ctypes.c_ulong)
    run_modal = message(ctypes.c_long)
    utf8 = message(ctypes.c_char_p)
    string_of = message(ctypes.c_void_p, ctypes.c_char_p)

    def text(value: str):
        return string_of(objc.objc_getClass(b"NSString"), sel(b"stringWithUTF8String:"), value.encode())

    pool = objc.objc_autoreleasePoolPush()
    try:
        panel = send(objc.objc_getClass(b"NSOpenPanel"), sel(b"openPanel"))
        if not panel:
            raise RuntimeError("no NSOpenPanel")
        send_bool(panel, sel(b"setCanChooseFiles:"), True)
        send_bool(panel, sel(b"setCanChooseDirectories:"), True)
        send_bool(panel, sel(b"setAllowsMultipleSelection:"), True)
        # a folder is a choice, not only a place to go into: the panel's *Open* takes it as it is
        send_bool(panel, sel(b"setTreatsFilePackagesAsDirectories:"), False)
        if title:
            send_object(panel, sel(b"setMessage:"), text(title))
        if start:
            url = send_object(objc.objc_getClass(b"NSURL"), sel(b"fileURLWithPath:"), text(start))
            if url:
                send_object(panel, sel(b"setDirectoryURL:"), url)
        if run_modal(panel, sel(b"runModal")) != _OK:
            return []
        urls = send(panel, sel(b"URLs"))
        paths = []
        for i in range(count_of(urls, sel(b"count"))):
            path = send(send_index(urls, sel(b"objectAtIndex:"), i), sel(b"path"))
            value = utf8(path, sel(b"UTF8String")) if path else None
            if value:
                paths.append(value.decode())
        return paths
    finally:
        objc.objc_autoreleasePoolPop(pool)
