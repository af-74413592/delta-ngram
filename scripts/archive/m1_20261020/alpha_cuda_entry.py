"""Resource-bounded launcher for the unchanged, frozen alpha training CLI."""
import os
from pathlib import Path
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
import torch

torch.cuda.set_per_process_memory_fraction(0.45, 0)
script = Path('/workspace/delta-ngram-alpha/scripts/train_alpha_ablation.py')
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
