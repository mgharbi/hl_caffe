# ----------------------------------------------------------------------------
# File:    settings.py
# Author:  Michael Gharbi <gharbi@mit.edu>
# Created: 2015-06-15
# ----------------------------------------------------------------------------
#
#
#
# ---------------------------------------------------------------------------#


import os.path as path
import os
import sys
import re

BASE_DIR   = path.split(path.dirname(path.abspath(__file__)))[0]

CAFFE_DIR    = path.join(BASE_DIR,"caffe")
DATA_DIR     = path.join(BASE_DIR,"data")
OUTPUT_DIR   = path.join(BASE_DIR,"output")
MODEL_DIR    = path.join(BASE_DIR,"trained_models")
DEBUG_DIR    = path.join(OUTPUT_DIR,"_debug",)
HL_TEMPLATES = path.join(BASE_DIR,"templates")
