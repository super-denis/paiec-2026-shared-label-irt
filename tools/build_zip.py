"""Build a PAIEC submission ZIP from a model.py:  python tools/build_zip.py model.py out.zip [extra_file ...]"""
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

model, out, extras = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
out.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(out, "w", compression=ZIP_DEFLATED) as z:
    for name, src in [("model.py", model)] + [(p.name, p) for p in extras]:
        e = ZipInfo(name); e.compress_type = ZIP_DEFLATED; e.external_attr = 0o100644 << 16
        z.writestr(e, src.read_bytes())
print("Wrote", out, out.stat().st_size, "bytes")
