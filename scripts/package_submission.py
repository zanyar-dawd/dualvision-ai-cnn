"""Embed verified trained weights and measured report assets in the required notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    assets = {}
    for name in ['mnist', 'cifar10']:
        checkpoint = ROOT / 'checkpoints' / 'baseline' / name / 'best.pt'
        output = ROOT / 'results' / 'baseline' / name
        training = json.loads((output / 'training.json').read_text(encoding='utf-8'))
        evaluation = json.loads((output / 'evaluation.json').read_text(encoding='utf-8'))
        assert training['training_status'] == 'complete' and not training['smoke']
        assert evaluation['source'] == 'torchvision-official' and evaluation['count'] == 10000
        assert evaluation['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selected = [checkpoint]
        selected.extend(p for p in output.rglob('*') if p.is_file() and p.suffix in {'.png', '.csv', '.json'}
                        and p.name not in {'split.json', 'artifact_manifest.json'})
        # Full split indices remain in the repository. They are not needed to review saved results.
        figure_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in selected
                         if p.parent == output and p.suffix in {'.png', '.csv', '.json'}}
        manifest_path = output / 'artifact_manifest.json'
        manifest_path.write_text(json.dumps(figure_hashes, indent=2), encoding='utf-8')
        selected.append(manifest_path)
        for path in selected:
            assets[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(assets.items()):
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True))
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    notebook = nbformat.read(ROOT / 'Homework_5_DualVision.ipynb', as_version=4)
    asset_cells = [cell for cell in notebook.cells if cell.cell_type == 'code' and 'SUBMISSION_ARCHIVE_B64 = ' in cell.source]
    assert len(asset_cells) == 1
    asset_cells[0].source = ('# Compact trained checkpoints and measured report assets; SHA-256 verified on restore.\n'
                             f'SUBMISSION_ARCHIVE_B64 = {encoded!r}\n'
                             'if SUBMISSION_ARCHIVE_B64 and not SMOKE_TEST and EXPERIMENT_ID == "baseline":\n'
                             '    restore_submission_assets(SUBMISSION_ARCHIVE_B64, PROJECT_ROOT)\n')
    asset_cells[0].metadata.update({'jupyter': {'source_hidden': True}, 'tags': ['embedded-artifacts']})
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / 'Homework_5_DualVision.ipynb')
    print(f'Embedded {len(assets)} real experiment assets ({len(buffer.getvalue()) / 1024**2:.2f} MiB compressed).')


if __name__ == '__main__':
    main()
