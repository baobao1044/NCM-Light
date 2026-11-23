"""
Build the distributable shaderpack zip.

The archive holds shaders/ at its root, which is what the game's shaderpack
loader expects when a pack is distributed as a zip rather than a folder.

Usage:
    python tools/package_pack.py <pack-folder> [output.zip]
"""

import os
import sys
import zipfile


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    pack = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else pack + ".zip"

    members = []
    for base, dirs, files in os.walk(pack):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(base, name)
            members.append((full, os.path.relpath(full, pack).replace(os.sep, "/")))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for full, arcname in members:
            archive.write(full, arcname)

    print("wrote %s (%d bytes)" % (out, os.path.getsize(out)))
    for _full, arcname in members:
        print("  " + arcname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
