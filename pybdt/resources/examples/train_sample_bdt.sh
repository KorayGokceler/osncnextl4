#!/usr/bin/env bash
# train_sample_bdt.sh

# Train a sample BDT.

# This is the training script included with pybdt
TRAIN_PY="$I3_SRC/pybdt/resources/scripts/train.py"
DATA_DIR="$I3_BUILD/pybdt/resources/examples/data/"
OUTPUT_DIR="$I3_BUILD/pybdt/resources/examples/output/"

# Create output directory
mkdir -p $OUTPUT_DIR

# Train with given training parameters:
$TRAIN_PY --num-trees 300 --depth 3 --beta .7 --prune-strength 35 \
    --frac-random-events=0.5 \
    --sig-weight="weight" \
    "a,b,c" $DATA_DIR/train_sig.ds $DATA_DIR/train_data.ds $OUTPUT_DIR/sample.bdt

if [ $? -ne '0' ]; then
    exit 1
fi

exit 0
