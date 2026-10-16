import base64
import hashlib
import io
import json
import zipfile
import pytest
from src.utilities.assets import restore_submission_assets


def archive_blob(name, data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps({name: hashlib.sha256(data).hexdigest()}))
    return base64.b64encode(buffer.getvalue()).decode()


def test_portable_assets_restore_and_conflicts(tmp_path):
    name = 'results/baseline/mnist/evidence.json'
    blob = archive_blob(name, b'{"measured": true}')
    restore_submission_assets(blob, tmp_path)
    restore_submission_assets(blob, tmp_path)
    (tmp_path / name).write_text('modified')
    with pytest.raises(ValueError, match='differs'):
        restore_submission_assets(blob, tmp_path)


def test_asset_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match='Invalid'):
        restore_submission_assets(archive_blob('../escape.txt', b'bad'), tmp_path)
