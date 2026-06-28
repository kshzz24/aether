import os
import tempfile
from pathlib import Path


def atomic_write(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(
        dir=p.parent, delete=False, mode="w", encoding="utf-8"
    )

    try:
        tmp.write(content)
        tmp.close()  #    must close before replacing
        os.replace(tmp.name, p)  # 3. atomic swap into place
    except BaseException:
        os.unlink(tmp.name)  #    clean up temp on failure
        raise
