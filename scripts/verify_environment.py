"""Run as: .venv/Scripts/python -m scripts.verify_environment."""
from pathlib import Path
import json
from src.utilities.runtime import seed_everything, select_device, write_json

if __name__ == '__main__':
    seed_everything()
    device, record = select_device()
    write_json(Path('results/environment.json'), record)
    print(json.dumps(record, indent=2))
