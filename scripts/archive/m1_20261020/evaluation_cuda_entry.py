"""Run one frozen evaluation with the registered auxiliary allocator cap."""
import os
from pathlib import Path
import runpy
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '8'
import torch

name = sys.argv.pop(1)
assert name in ('evaluate_alpha_checkpoint.py', 'evaluate_recent_news_continuation.py')
torch.cuda.set_per_process_memory_fraction(0.45, 0)
script = Path('/workspace/delta-ngram-evaluation-20261019/scripts') / name
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name='__main__')
