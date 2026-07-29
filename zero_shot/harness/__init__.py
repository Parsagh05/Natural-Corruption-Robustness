# Robustness Evaluation Harness for Contrastive VLMs in Anomaly Detection
"""
Zero-shot robustness evaluation harness using Hendrycks corruptions.
Designed to run inside a Kaggle Notebook environment with VSCode Tunnel support.
"""

import sys
from pathlib import Path


# Keep the historical ``from harness...`` entry point working when commands
# are launched from zero_shot/, while shared assets live one directory above.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
