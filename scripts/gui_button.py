import os
import re
import subprocess
import sys
from pathlib import Path

import traffic

ICON_PATH = Path(traffic.__file__).parent.parent / "icons"


class LinuxApp:

    linux_desktop = """#!/usr/bin/env xdg-open
[Desktop Entry]
Version=1.0
Name=traffic
Comment=A GUI for manipulating and analysing air traffic data
Exec={script_path}
Icon={icon_path}
Terminal=false
Type=Application
Categories=Utility;"""

    linux_script = """#!/usr/bin/env bash
{python_exe} -m traffic gui
"""

    def detect_environment(self):
        # TODO if 'conda' in sys.exec_prefix for conda root
        # but how to detect whether no conda environment?
        if hasattr(sys, "real_prefix"):  # ok with virtualenv
            return "source {}/bin/activate".format(sys.exec_prefix)
        for e in sys.path:
            match = re.match("(.*)/envs/(.*)/lib/python3.\d", e)  # noqa: W605
            if match:
                prefix = 'export PATH="{}:$PATH"\n'.format(match.group(1))
                if sys.platform == "linux":
                    prefix += 'export LD_LIBRARY_PATH="{}:$LD_LIBRARY_PATH"\n'
                    prefix = prefix.format(match.group(0)[:-9])
                return prefix + "source {}/bin/activate {}".format(
                    sys.exec_prefix, match.group(2)
                )

    def make_app(self):
        icon_path = ICON_PATH / "travel-white.svg"
        script_path = os.path.expanduser("~/.local/bin/traffic_gui.sh")

        with open(script_path, "w") as fh:
            fh.write(self.linux_script.format(python_exe=sys.executable))
            mode = os.fstat(fh.fileno()).st_mode
            mode |= 0o111
            os.fchmod(fh.fileno(), mode & 0o7777)

        with open("traffic.desktop", "w") as fh:
            fh.write(
                self.linux_desktop.format(
                    icon_path=icon_path, script_path=script_path
                )
            )
            mode = os.fstat(fh.fileno()).st_mode
            mode |= 0o111
            os.fchmod(fh.fileno(), mode & 0o7777)

        subprocess.call(
            "desktop-file-install --dir ~/.local/share/applications "
            "traffic.desktop --rebuild-mime-info-cache",
            shell=True,
        )


class DarwinApp:

    darwin_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN"
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>LSBackgroundOnly</key>
    <string>0</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleName</key>
    <string>traffic</string>
    <key>CFBundleExecutable</key>
    <string>MacOS/main.sh</string>
    <key>CFBundleGetInfoString</key>
    <string>A GUI for manipulating and analysing air traffic data</string>
    <key>CFBundleIconFile</key>
    <string>travel.icns</string>
    <key>CFBundleIdentifier</key>
    <string>org.xoolive.traffic</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleDisplayName</key>
    <string>traffic</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2017, Xavier Olive</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""

    darwin_script = """#! /bin/bash

DIR=${0%/*}
${DIR}/traffic -c "from traffic.console import gui; gui.main(None)"
"""

    def make_app(self):
        pythonapp = sys.exec_prefix

        if not hasattr(sys, "real_prefix"):  # means no virtualenv
            pythonapp = os.path.join(pythonapp, "Resources")

        # if "conda" in sys.version or "Continuum" in sys.version:
        if (Path(sys.prefix) / "conda-meta").exists():
            pythonapp = sys.executable
        else:
            pythonapp = os.path.join(
                pythonapp, "Python.app", "Contents", "MacOS", "Python"
            )

        os.makedirs(os.path.join("traffic.app", "Contents", "MacOS"))
        os.makedirs(os.path.join("traffic.app", "Contents", "Resources"))

        if "conda" in sys.version or "Continuum" in sys.version:
            pass
        else:
            os.link(
                pythonapp,
                os.path.join("traffic.app", "Contents", "MacOS", "traffic"),
            )

        with open(
            os.path.join("traffic.app", "Contents", "Info.plist"), "w"
        ) as fh:
            fh.write(self.darwin_plist)

        with open(
            os.path.join("traffic.app", "Contents", "MacOS", "main.sh"), "w"
        ) as fh:
            if "conda" in sys.version or "Continuum" in sys.version:
                linuxapp = LinuxApp()
                fh.write(
                    linuxapp.linux_script.format(python_exe=sys.executable)
                )
            else:
                fh.write(self.darwin_script)
            mode = os.fstat(fh.fileno()).st_mode
            mode |= 0o111
            os.fchmod(fh.fileno(), mode & 0o7777)

        icon_path = ICON_PATH / "travel.icns"

        os.link(
            icon_path,
            os.path.join("traffic.app", "Contents", "Resources", "travel.icns"),
        )


class WindowsApp:
    windows_batch = """"{}" -x %0 %*    &goto :eof
from traffic.console import gui;
gui.main(None)
"""

    def make_app(self):

        with open("traffic.bat", "w") as fh:
            fh.write(self.windows_batch.format(sys.executable))

        try:
            from win32com.client import Dispatch
        except ImportError:
            subprocess.call("conda install pywin32")
            print("Missing package installed, relaunch script")
            return

        path = "traffic.lnk"
        target = str(Path("traffic.bat").absolute())
        icon_path = ICON_PATH / "travel.ico"

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.TargetPath = target
        shortcut.IconLocation = icon_path.as_posix()
        shortcut.save()


def main():

    fname = {"linux": LinuxApp(), "darwin": DarwinApp(), "win32": WindowsApp()}

    app = fname.get(sys.platform, None)
    if app is not None:
        app.make_app()  # type: ignore


if __name__ == "__main__":
    main()
