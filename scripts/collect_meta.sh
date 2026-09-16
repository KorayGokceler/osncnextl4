#!/bin/bash
# Copy the pass2 source of record into the repository for reading.
# Only text sources -- no binaries, no models, no data.
set -u
M=/cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-07
F=/data/sim/DeepCore/2018/workspace/fridge/processing/samples/oscNext/selection/level4
OUT=reference/oscNext_meta

mkdir -p $OUT/{oscNext,tau_bdt,event_selection,fridge_level4}

# 1. the LIVE L4/L3/L5 selection scripts (ours is the commented-out copy)
cp -v $M/src/oscNext/python/selection/*.py            $OUT/oscNext/       2>/dev/null
cp -v $M/src/oscNext/python/tools/*.py                $OUT/oscNext/       2>/dev/null

# 2. VICH
cp -rv $M/src/tau_bdt/private $M/src/tau_bdt/public   $OUT/tau_bdt/       2>/dev/null

# 3. accumulated_time / separation_in_cogs
cp -rv $M/src/analysis/python/event_selection         $OUT/event_selection/ 2>/dev/null

# 4. the training recipe
cp -v $F/*.py                                         $OUT/fridge_level4/ 2>/dev/null

# no bytecode, no binaries
find $OUT -name "*.pyc" -delete
find $OUT -type f -size +2M -print -delete

echo
echo "=== total ==="
du -sh $OUT
find $OUT -type f | wc -l
