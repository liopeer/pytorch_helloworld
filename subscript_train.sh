#!/bin/bash
source /scratch_net/biwidl311/peerli/conda/etc/profile.d/conda.sh
conda activate liotorch
python -u train.py "$@"