#!/usr/bin/env python
# ----------------------------------------------------------------------------
# File:    hl_caffe.py
# Author:  Michael Gharbi <gharbi@mit.edu>
# Created: 2015-08-18
# ----------------------------------------------------------------------------
#
#
#
# ---------------------------------------------------------------------------#


import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "python"))

import scripts
from settings import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Main interface with HLCaffe")
    subparsers = parser.add_subparsers()

    # Sub-commands
    parser_run         = subparsers.add_parser('run')
    parser_show         = subparsers.add_parser('show')
    parser_gen         = subparsers.add_parser('gen')

    # ---------------------------------------------------------------------------

    # Run
    parser_run.set_defaults(func = scripts.run)
    parser_run.add_argument("model")
    parser_run.add_argument("--use-cpu", action='store_true')

    # Show
    parser_show.set_defaults(func = scripts.show)
    parser_show.add_argument("model")

    # Gen
    parser_gen.set_defaults(func = scripts.gen)
    parser_gen.add_argument("model")

    # ---------------------------------------------------------------------------

    # Parse and run
    args = parser.parse_args()
    args.func(args)
