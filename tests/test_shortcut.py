"""The desktop icon `smappy-gui --desktop-icon` makes (`smappy.shortcut`)."""
import struct
import sys

import pytest

from smappy import shortcut


def test_the_icon_file_holds_every_size_up_to_256_as_png():
    data = (shortcut.ICONS / "smappy.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind) == (0, 1) and count >= 4
    sides = []
    for i in range(count):
        side, _, _, _, _, _, size, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert data[offset:offset + 8] == b"\x89PNG\r\n\x1a\n" and offset + size <= len(data)
        sides.append(side or 256)
    assert {16, 32, 48, 256} <= set(sides)


def test_the_windows_shortcut_starts_this_python_with_the_gui_and_smappys_icon():
    script = shortcut.windows_script(r"C:\it's\python.exe", shortcut.ICONS / "smappy.ico")
    assert "GetFolderPath('Desktop')" in script                  # where Windows says, OneDrive or not
    assert r"'C:\it''s\python.exe'" in script                     # quoted for PowerShell
    assert "'-m smappy.cli.gui'" in script and "smappy.ico,0'" in script


@pytest.mark.skipif(sys.platform != "win32", reason="the shell's shortcut object is Windows'")
def test_on_windows_the_shortcut_is_made_and_reads_back(tmp_path):
    import subprocess
    made = shortcut.make(tmp_path)
    assert made == tmp_path / "SMAPpy.lnk" and made.is_file()
    read = subprocess.run(["powershell", "-NoProfile", "-Command",
                           f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{made}'); "
                           "$s.TargetPath; $s.IconLocation"], capture_output=True, text=True).stdout.split("\n")
    assert read[0].strip().lower() == sys.executable.lower() and read[1].strip().endswith("smappy.ico,0")


@pytest.mark.skipif(sys.platform != "linux", reason="a freedesktop.org launcher")
def test_on_linux_the_launcher_goes_in_the_menu_and_on_the_desktop(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    (tmp_path / "Desktop").mkdir()
    made = shortcut.make(tmp_path)
    assert made == tmp_path / "Desktop" / "smappy.desktop"
    entry = (tmp_path / ".local" / "share" / "applications" / "smappy.desktop").read_text()
    assert f'Exec="{sys.executable}" -m smappy.cli.gui %F' in entry and "smappy_256.png" in entry
