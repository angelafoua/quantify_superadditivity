# -*- coding: utf-8 -*-
"""quantify_superadditivity.ipynb

Colab notebook for running the superadditivity experiments.

Original file is located at
    https://colab.research.google.com/drive/1q1_Evrhzlb2Jg4bJ_8_ehkwAFEZI5Pij

Usage:
    Open in Google Colab with an A100 GPU runtime, then Run All.
    On the first run, datasets download from the internet and are cached
    on Google Drive so subsequent sessions skip the download entirely.
"""

import torch
assert torch.cuda.is_available(), 'No GPU. Runtime -> Change runtime type -> A100 GPU.'
print('GPU:', torch.cuda.get_device_name(0))

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import os
if not os.path.isdir('/content/quantify_superadditivity'):
    !git clone https://github.com/angelafoua/quantify_superadditivity /content/quantify_superadditivity
# %cd /content/quantify_superadditivity
!git checkout main && git pull origin main

!pip install -q hydra-core==1.3.2 omegaconf==2.3.0 h5py networkx statsmodels seaborn tabulate tqdm scikit-learn pandas scipy
!pip install -q -e . --no-deps
!python scripts/validate_setup.py

# --- Cache datasets on Google Drive to avoid re-downloading each session ---
import os

DRIVE_DATA = '/content/drive/MyDrive/superadditivity_data'
LOCAL_DATA = '/content/quantify_superadditivity/data'

os.makedirs(DRIVE_DATA, exist_ok=True)

if os.path.isdir(LOCAL_DATA) and not os.path.islink(LOCAL_DATA):
    # First run: move already-downloaded data to Drive before symlinking
    !cp -rn {LOCAL_DATA}/* {DRIVE_DATA}/ 2>/dev/null || true
    !rm -rf {LOCAL_DATA}

if os.path.islink(LOCAL_DATA):
    os.remove(LOCAL_DATA)

os.symlink(DRIVE_DATA, LOCAL_DATA)
print('data ->', os.path.realpath(LOCAL_DATA))

# --- Cache outputs on Google Drive ---
DRIVE_OUT = '/content/drive/MyDrive/superadditivity_outputs'
os.makedirs(DRIVE_OUT, exist_ok=True)
if os.path.islink('outputs') or os.path.exists('outputs'):
    !rm -rf outputs
os.symlink(DRIVE_OUT, 'outputs')
print('outputs ->', os.path.realpath('outputs'))

# --- Quick validation (1 seed, 4 runs) ---
!python scripts/run_sweep.py --experiment core_factorial_quick --resume

# --- Full factorial (5 run_seeds x 3 graph_seeds = 15 runs/cell, 60 total) ---
!python scripts/run_sweep.py --experiment core_factorial --resume

# --- Analysis & visualisation for both experiments ---
!python scripts/run_analysis.py --results_dir outputs/core_factorial_quick
!python scripts/run_visualization.py --results_dir outputs/core_factorial_quick

!python scripts/run_analysis.py --results_dir outputs/core_factorial
!python scripts/run_visualization.py --results_dir outputs/core_factorial
