"""Operational constants for VoxelScribe."""

# Paths
DATA_DIR = "data/raw"
CT_FILE = "ctlayer_32.tif"
INK_FILE = "inklabels.png"
RESULTS_DIR = "results"

# Region of interest (crop)
# Picked to land inside the papyrus body with a balanced mix of ink / no-ink
# pixels in both the training (top) and testing (bottom) halves of the ROI.
CROP_TOP = 1600
CROP_LEFT = 2200
CROP_H = 300
CROP_W = 300

# Reproducibility
SEED = 42

# Default sampling hyperparameters
N_BURN = 200
N_SAMPLE = 500
ENERGY_LOG_EVERY = 10
