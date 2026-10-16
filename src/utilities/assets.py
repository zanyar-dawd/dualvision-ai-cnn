"""Restore compact, checksum-verified report/checkpoint assets from the notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


def restore_submission_assets(encoded_archive: str, destination: Path) -> None:
    """Restore only declared project assets; refuse traversal and conflicting files."""
    destination = Path(destination).resolve()
    archive_bytes = base64.b64decode(encoded_archive, validate=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if set(archive.namelist()) != set(manifest) | {'manifest.json'}:
            raise ValueError('Unexpected archive entries.')
        for name, digest in manifest.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('Invalid submission asset path.')
            if not (name.startswith('checkpoints/baseline/') or name.startswith('results/baseline/')):
                raise ValueError('Submission archive contains an unrelated path.')
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Submission asset escapes the working directory.')
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f'Corrupt submission asset: {name}')
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Existing artifact differs: {name}. Open this notebook in a fresh folder.')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print('Verified embedded trained checkpoints and report assets; no dataset download or training is required.')
