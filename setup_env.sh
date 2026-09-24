#!/bin/bash
#
# Find, verify and enter the IceTray environment.
#
#   ./setup_env.sh            -> report what was found (changes nothing;
#                                also: ./setup_env.sh find)
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
    if [ -n "$OSCNEXT_I3_BUILD" ]; then
        if [ -x "$OSCNEXT_I3_BUILD/env-shell.sh" ]; then
            echo "$OSCNEXT_I3_BUILD/env-shell.sh"; return 0
        fi
        echo "[!] OSCNEXT_I3_BUILD=$OSCNEXT_I3_BUILD holds no executable" \
             "env-shell.sh -- ignored, searching elsewhere" >&2
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
    # 4) cvmfs metaproject: the one this pipeline was verified on first.  A
    #    plain `sort -r` over every toolset picked the wrong platform and,
    #    comparing "v1.9.2" with "v1.17.0" as strings, an older release.
    local verified="$CVMFS/py3-v4.4.2/${OS_ARCH:-RHEL_9_x86_64_v2}/metaprojects/icetray/v1.17.0/env-shell.sh"
    [ -x "$verified" ] && { echo "$verified"; return 0; }
    for p in $(ls -d "$CVMFS"/py3-v*/*/metaprojects/icetray/*/env-shell.sh 2>/dev/null | sort -rV); do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}

# A cvmfs metaproject's env-shell.sh expects its toolset to be loaded first
# (eval $(.../py3-vX/setup.sh), which sets SROOT and the python it was built
# with).  Do that when it has not been done.
load_toolset_for() {
    local es="$1" root
    [ -n "${SROOT:-}" ] && return 0
    root="$(echo "$es" | grep -oE "^$CVMFS/py3-v[0-9.]+")" || return 0
    if [ -x "$root/setup.sh" ]; then
        echo "-> loading the toolset first: eval \$($root/setup.sh)" >&2
        eval "$("$root/setup.sh")"
    fi
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
    if already_inside; then
        echo "icetray is ALREADY loaded here ($(python -c 'import sys; print(sys.executable)')):"
        echo "run / kernel / lab use THIS environment, not the candidates below."
        echo
    fi

    echo "env-shell.sh candidates found:"
    local any=0
    for p in /data/user/"$(whoami)"/icetray_build/build/env-shell.sh \
             /data/user/"$(whoami)"/*/build/env-shell.sh \
             /data/user/"$(whoami)"/build/env-shell.sh \
             "$HOME"/icetray/build/env-shell.sh "$HOME"/*/build/env-shell.sh \
             "$HOME"/*/*/build/env-shell.sh "$HOME"/build/env-shell.sh; do
        [ -x "$p" ] && { echo "  [own build   ] $p"; any=1; }
    done
    for p in $(ls -d "$CVMFS"/py3-v*/*/metaprojects/icetray/*/env-shell.sh 2>/dev/null | sort -rV | head -5); do
        echo "  [cvmfs       ] $p"; any=1
    done
    [ "$any" = 0 ] && echo "  (none found)"
    echo

    local es
    if already_inside; then
        echo "icecube + lightgbm import test (this environment):"
        python -c "import sys, icecube, lightgbm; from icecube.icetray import I3Tray; print('python  :', sys.executable); print('icecube :', icecube.__file__); print('lightgbm:', lightgbm.__version__)" 2>&1 | sed 's/^/  /'
    elif es="$(find_env_shell)"; then
        echo "Will use     : $es"
        local bd; bd="$(dirname "$es")"
        local ts; ts="$(detect_toolset "$bd")"
        [ -n "${ts:-}" ] && echo "Build toolset (CMakeCache): $ts"
        echo
        echo "icecube + lightgbm import test:"
        load_toolset_for "$es"
        "$es" -- python -c "import sys, icecube, lightgbm; from icecube.icetray import I3Tray; print('python  :', sys.executable); print('icecube :', icecube.__file__); print('lightgbm:', lightgbm.__version__)" 2>&1 | sed 's/^/  /'
    else
        echo "env-shell.sh NOT FOUND."
        echo "If your own build is somewhere else:"
        echo "    export OSCNEXT_I3_BUILD=/full/path/build"
    fi
    echo "=================================================================="
}

# Is an icetray environment ALREADY loaded?
#
# WHY THIS EXISTS: env-shell.sh refuses to run inside a different build --
#
#     I3_BUILD CHANGED
#     It appears that you are attempting to load an icetray environment
#     different than the one already loaded
#
# and that is exactly what happened: a shell opened from the cvmfs metaproject
# has I3_BUILD pointing at .../share/icetray, which holds no env-shell.sh, so
# find_env_shell fell past it to a LOCAL build under /data/user and tried to
# load that instead.  Two different environments depending on how you started.
#
# When icetray already imports there is nothing to load: run the command here.
already_inside() {
    python -c "import icecube" >/dev/null 2>&1
}

run_here_or_in_shell() {
    [ $# -gt 0 ] || { echo "run: no command given (./setup_env.sh run <cmd> ...)"; exit 2; }
    if already_inside; then
        echo "-> icetray is already loaded ($(python -c "import sys; print(sys.executable)")); running directly" >&2
        exec "$@"
    fi
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    load_toolset_for "$es"
    exec "$es" -- "$@"
}

case "${1:-report}" in
  report|find)
    report
    ;;
  shell)
    if already_inside; then
        echo "icetray is already loaded here:"
        python -c "import sys, icecube.icetray as i; print('  python :', sys.executable); print('  icetray:', i.__file__)"
        echo "Opening a second one would be a DIFFERENT environment.  Nothing to do."
        exit 0
    fi
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    echo "-> $es"
    load_toolset_for "$es"
    exec "$es"
    ;;
  run)
    shift
    run_here_or_in_shell "$@"
    ;;
  kernel)
    # Register the Jupyter kernel FROM INSIDE ICETRAY.  This is the only
    # correct way for the notebook to see icetray: the python in kernel.json
    # must be the python inside env-shell.
    if already_inside; then
        python -m ipykernel install --user \
            --name icetray --display-name "IceTray (oscNext L4)" \
          && echo "Registered from the environment already loaded."
        exit $?
    fi
    es="$(find_env_shell)" || { echo "env-shell.sh not found"; exit 1; }
    load_toolset_for "$es"
    "$es" -- python -m ipykernel install --user \
        --name icetray --display-name "IceTray (oscNext L4)" \
      && echo "Registered.  In the notebook: Kernel > Change Kernel > 'IceTray (oscNext L4)'"
    ;;
  lab)
    shift
    run_here_or_in_shell jupyter lab --no-browser --port "${1:-8888}"
    ;;
  *)
    sed -n '2,/^$/p' "$0"
    exit 1
    ;;
esac
