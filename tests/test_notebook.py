import ast
import json
from pathlib import Path
import nbformat
from scripts.build_notebook import inline_module


def test_notebook_is_valid_self_contained_and_has_required_sections():
    notebook = nbformat.read('Homework_5_DualVision.ipynb', as_version=4)
    nbformat.validate(notebook)
    markdown = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'markdown')
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    for section in ['Abstract', 'Introduction', 'objectives', 'Dataset descriptions', 'MNIST methodology',
                    'MNIST implementation', 'MNIST experimental results', 'CIFAR-10 methodology',
                    'CIFAR-10 implementation', 'CIFAR-10 experimental results', 'error analysis',
                    'Comparative discussion', 'Limitations', 'Conclusion', 'References']:
        assert section in markdown
    assert 'from src.' not in code and 'import src' not in code
    ast.parse(code)
    for name in ['DigitNet', 'VisionNet', 'epoch_pass', 'fit', 'grad_cam', 'predict_file', 'evaluate_once']:
        assert f'class {name}' in code or f'def {name}' in code
    for source in Path('src').rglob('*.py'):
        if source.name != '__init__.py':
            assert inline_module(str(source)).strip() in code
    assert 'ALLOW_TRAINING = False' in code and 'DOWNLOAD_DATA = False' in code
