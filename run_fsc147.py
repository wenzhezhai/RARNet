"""Edit the three paths, then run: python run_fsc147.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'data' / 'FSC147'
CHECKPOINT = ROOT / 'weights' / 'rarnet.pth'
OUTPUT_ROOT = ROOT / 'outputs' / 'fsc147'

if __name__ == '__main__':
    from inference.runner import run
    run(DATA_ROOT, CHECKPOINT, OUTPUT_ROOT, splits=('test',))
