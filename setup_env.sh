#!/bin/bash
#
# Find, verify and enter the IceTray environment.
#
#   ./setup_env.sh            -> report what was found (changes nothing)
#   ./setup_env.sh shell      -> open the icetray shell
#   ./setup_env.sh run <cmd>  -> run one command inside icetray
#   ./setup_env.sh kernel     -> register an "IceTray" kernel with Jupyter
#   ./setup_env.sh lab        -> start jupyter lab inside icetray
#
# WHY THIS SCRIPT EXISTS:
#   env-shell.sh OPENS A NEW SHELL.  Writing, as the README once did,
#       eval $(setup.sh)
#       env-shell.sh
#       python scripts/diagnose_env.py    <-- THIS LINE DOES NOT WORK
#   runs the last line AFTER leaving env-shell, i.e. without icetray.  It is
#   the number one cause of "icetray will not import".
#   For a single command:  env-shell.sh -- python scripts/diagnose_env.py

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVMFS=/cvmfs/icecube.opensciencegrid.org

# --- can also be given by hand through OSCNEXT_I3_BUILD ----------------------
: "${OSCNEXT_I3_BUILD:=}"

find_env_shell() {
    # 1) Given by hand
    if [ -n "$OSCNEXT_I3_BUILD" ] && [ -x "$OSCNEXT_I3_BUILD/env-shell.sh" ]; then
        echo "$OSCNEXT_I3_BUILD/env-shell.sh"; return 0
    fi
    # 2) An already defined I3_BUILD
    if [ -n "${I3_BUILD:-}" ] && [ -x "$I3_BUILD/env-shell.sh" ]; then
        echo "$I3_BUILD/env-shell.sh"; return 0
    fi
    # 3) Self-compiled builds.
    #    /data/user/$USER is the likeliest place: home is quota-limited on
    #    cobalt, so builds go there (see README).
    for p in /data/user/"$(whoami)"/icetray_build/build/env-shell.sh \
             /data/user/"$(whoami)"/*/build/env-shell.sh \
             /data/user/"$(whoami)"/build/env-shell.sh \
             "$HOME"/icetray/build/env-shell.sh \
             "$HOME"/*/build/env-shell.sh \
             "$HOME"/*/*/build/env-shell.sh \
             "$HOME"/build/env-shell.sh; do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    # 4) cvmfs metaproject
    for p in $(ls -d "$CVMFS"/py3-v*/*/metaprojects/icetray/*/env-shell.sh 2>/dev/null | sort -r); do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}

# For a self-compiled build: find which cvmfs python it was built against.
# Running it under the wrong py3-vX gives "undefined symbol" / import failures.
detect_toolset() {
    local build_dir="$1"
    local cache="$build_dir/CMakeCache.txt"
    [ -f "$cache" ] || return 1
    grep -oE "$CVMFS/py3-v[0-9.]+/[A-Za-z0-9_]+" "$cache" 2>/dev/null \
        | head -1
}

report() {
    echo "=================================================================="
    echo "ICETRAY ENVIRONMENT REPORT"
    echo "=================================================================="
    echo "repo      : $HERE"
    echo "SROOT     : ${SROOT:-<unset>}"
    echo "I3_BUILD  : ${I3_BUILD:-<unset>}"
    echo

    echo "env-shell.sh candidates found:"
    local any=0
    for p in /data/user/"$(whoami)"/icetray_build/build/env-shell.sh \
             /data/user/"$(whoami)"/*/build/env-shell.sh \
             /data/user/"$(whoami)"/build/env-shell.sh \
             "$HOME"/icetray/build/env-shell.sh "$HOME"/*/build/env-shell.sh \
             "$HOME"/*/*/build/env-shell.sh "$HOME"/build/env-shell.sh; do
        [ -x "$p" ] && { echo "  [own build   ] $p"; any=1; }
    done
    for p in $(ls -d "$CVMFS"/py3-v*/*/metaprojects/icetray/*/env-shell.sh 2>/dev/null | sort -r | head -5); do
        echo "  [cvmfs       ] $p"; any=1
    done
    [ "$any" = 0 ] && echo "  (none found)"
    echo

    local es
    if es="$(find_env_shell)"; then
        echo "Will use     : $es"
        local bd; bd="$(dirname "$es")"
        local ts; ts="$(detect_toolset "$bd")"
        [ -n "${ts:-}" ] && echo "Build toolset (CMakeCache): $ts"
        echo
        echo "icecube + lightgbm import test:"
        "$es" -- python "$HERE/oscnext_l4/env.py" 2>&1 | sed 's/^/  /'
    else
        echo "env-shell.sh NOT FOUND."
        echo "If your own build is somewhere else:"
        echo "    export OSCNEXT_I3_BUILD=/full/path/build"
    fi
    echo "=================================================================="
}

case "${1:-report}" in
  report)
    report
    ;;
  shell)
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    echo "-> $es"
    exec "$es"
    ;;
  run)
    shift
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    exec "$es" -- "$@"
    ;;
  kernel)
    # Register the Jupyter kernel FROM INSIDE ICETRAY.  This is the only
    # correct way for the notebook to see icetray: the python in kernel.json
    # must be the python inside env-shell.
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    "$es" -- python -m ipykernel install --user \
        --name icetray --display-name "IceTray (oscNext L4)" \
      && echo "Registered.  In the notebook: Kernel > Change Kernel > 'IceTray (oscNext L4)'"
    ;;
  lab)
    shift
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    exec "$es" -- jupyter lab --no-browser --port "${1:-8888}"
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
