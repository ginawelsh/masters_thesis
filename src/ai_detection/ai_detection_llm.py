# LLM AI detection workflows

# tool used: GPT

import os
import glob

venv_paths = glob.glob(os.path.join(os.path.expanduser("~"), "*env*"))
for venv in venv_paths:
    if os.path.isdir(venv) and os.path.exists(os.path.join(venv, "bin", "activate")):
        print(venv)