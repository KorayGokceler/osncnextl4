#!/bin/bash
# Copy the pass2 source of record into the repository for reading.
# Only text sources -- no binaries, no models, no data.
#
# The two rewritten modules' sources are the ones CLAUDE.md cites:
#   VICH             src/tau-bdt/python/I3CutL7Module.py   (hyphen in the
#                    directory, underscore in the import)
#   accumulated_time src/analysis/private/analysis/event_selection/
#                    CalculateVariables.cxx (+ Variables.h)
# A path that is not there is reported, not silently skipped.
set -u
cd "$(dirname "$0")/.." || exit 1          # repo root, whatever the cwd
M=/cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-07
F=/data/sim/DeepCore/2018/workspace/fridge/processing/samples/oscNext/selection/level4
OUT=reference/oscNext_meta

mkdir -p "$OUT"/{oscNext,tau_bdt,event_selection,fridge_level4}

missing=0
copy() {   # copy <dest> <source>...  -- say what is not there
    local dest="$1"; shift
    local src
    for src in "$@"; do
        if [ -e "$src" ]; then
            cp -r "$src" "$dest"/
        else
            echo "[!] not found: $src"; missing=1
        fi
    done
}

# 1. the LIVE L4/L3/L5 selection scripts (ours is the commented-out copy)
copy "$OUT/oscNext" "$M"/src/oscNext/python/selection/*.py "$M"/src/oscNext/python/tools/*.py

# 2. VICH
copy "$OUT/tau_bdt" "$M"/src/tau-bdt/python

# 3. accumulated_time / separation_in_cogs
copy "$OUT/event_selection" "$M"/src/analysis/private/analysis/event_selection \
    "$M"/src/analysis/public/analysis/event_selection

# 4. the training recipe
copy "$OUT/fridge_level4" "$F"/*.py

# no bytecode, no binaries
find "$OUT" -name "*.pyc" -delete
find "$OUT" -type f -size +2M -print -delete

echo
echo "=== total ==="
du -sh "$OUT"
find "$OUT" -type f | wc -l
[ "$missing" = 0 ] || echo "[!] some sources were not found -- see above"
