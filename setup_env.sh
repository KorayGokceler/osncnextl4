#!/bin/bash
#
# IceTray ortamini bul, dogrula, ac.
#
#   ./setup_env.sh            -> ne bulundugunu raporla (hicbir sey degistirmez)
#   ./setup_env.sh shell      -> icetray shell'ini ac
#   ./setup_env.sh run <cmd>  -> tek komutu icetray icinde calistir
#   ./setup_env.sh kernel     -> Jupyter'a "IceTray" kernel'ini kaydet
#   ./setup_env.sh lab        -> jupyter lab'i icetray icinde baslat
#
# NEDEN BU SCRIPT VAR:
#   env-shell.sh YENI BIR SHELL acar.  README'deki gibi
#       eval $(setup.sh)
#       env-shell.sh
#       python diagnose_env.py        <-- BU SATIR CALISMAZ
#   yazarsaniz son satir env-shell'den CIKTIKTAN sonra calisir, yani icetray
#   olmayan ortamda.  "icetray import edilmiyor" hatasinin 1 numarali sebebi.
#   Tek komut icin dogrusu:  env-shell.sh -- python diagnose_env.py

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVMFS=/cvmfs/icecube.opensciencegrid.org

# --- OSCNEXT_I3_BUILD ile elle de verilebilir --------------------------------
: "${OSCNEXT_I3_BUILD:=}"

find_env_shell() {
    # 1) Elle verilen
    if [ -n "$OSCNEXT_I3_BUILD" ] && [ -x "$OSCNEXT_I3_BUILD/env-shell.sh" ]; then
        echo "$OSCNEXT_I3_BUILD/env-shell.sh"; return 0
    fi
    # 2) Zaten tanimli I3_BUILD
    if [ -n "${I3_BUILD:-}" ] && [ -x "$I3_BUILD/env-shell.sh" ]; then
        echo "$I3_BUILD/env-shell.sh"; return 0
    fi
    # 3) Kendi derlediginiz build'ler
    #    /data/user/$USER en olasi yer: cobalt'ta home kotali oldugu icin
    #    build oraya yapiliyor (bkz. README).
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

# Kendi derlenmis build ise: hangi cvmfs python'u ile derlendigini bul.
# Yanlis py3-vX ile calistirmak "undefined symbol" / "import edilmiyor" verir.
detect_toolset() {
    local build_dir="$1"
    local cache="$build_dir/CMakeCache.txt"
    [ -f "$cache" ] || return 1
    grep -oE "$CVMFS/py3-v[0-9.]+/[A-Za-z0-9_]+" "$cache" 2>/dev/null \
        | head -1
}

report() {
    echo "=================================================================="
    echo "ICETRAY ORTAM RAPORU"
    echo "=================================================================="
    echo "repo      : $HERE"
    echo "SROOT     : ${SROOT:-<bos>}"
    echo "I3_BUILD  : ${I3_BUILD:-<bos>}"
    echo

    echo "Bulunan env-shell.sh adaylari:"
    local any=0
    for p in /data/user/"$(whoami)"/icetray_build/build/env-shell.sh \
             /data/user/"$(whoami)"/*/build/env-shell.sh \
             /data/user/"$(whoami)"/build/env-shell.sh \
             "$HOME"/icetray/build/env-shell.sh "$HOME"/*/build/env-shell.sh \
             "$HOME"/*/*/build/env-shell.sh "$HOME"/build/env-shell.sh; do
        [ -x "$p" ] && { echo "  [kendi build ] $p"; any=1; }
    done
    for p in $(ls -d "$CVMFS"/py3-v*/*/metaprojects/icetray/*/env-shell.sh 2>/dev/null | sort -r | head -5); do
        echo "  [cvmfs       ] $p"; any=1
    done
    [ "$any" = 0 ] && echo "  (hicbiri bulunamadi)"
    echo

    local es
    if es="$(find_env_shell)"; then
        echo "Kullanilacak : $es"
        local bd; bd="$(dirname "$es")"
        local ts; ts="$(detect_toolset "$bd")"
        [ -n "${ts:-}" ] && echo "Derleme toolset (CMakeCache): $ts"
        echo
        echo "icecube + lightgbm import testi:"
        "$es" -- python "$HERE/icetray_env.py" 2>&1 | sed 's/^/  /'
    else
        echo "env-shell.sh BULUNAMADI."
        echo "Kendi build'iniz baska bir yerdeyse:"
        echo "    export OSCNEXT_I3_BUILD=/tam/yol/build"
    fi
    echo "=================================================================="
}

case "${1:-report}" in
  report)
    report
    ;;
  shell)
    es="$(find_env_shell)" || { echo "env-shell.sh bulunamadi"; exit 1; }
    echo "-> $es"
    exec "$es"
    ;;
  run)
    shift
    es="$(find_env_shell)" || { echo "env-shell.sh bulunamadi"; exit 1; }
    exec "$es" -- "$@"
    ;;
  kernel)
    # Jupyter kernel'ini ICETRAY ICINDEN kaydet.  Notebook'un icetray'i
    # gorebilmesinin TEK dogru yolu bu: kernel.json icindeki python,
    # env-shell icindeki python olmali.
    es="$(find_env_shell)" || { echo "env-shell.sh bulunamadi"; exit 1; }
    "$es" -- python -m ipykernel install --user \
        --name icetray --display-name "IceTray (oscNext L4)" \
      && echo "Kaydedildi.  Notebook'ta: Kernel > Change Kernel > 'IceTray (oscNext L4)'"
    ;;
  lab)
    shift
    es="$(find_env_shell)" || { echo "env-shell.sh bulunamadi"; exit 1; }
    exec "$es" -- jupyter lab --no-browser --port "${1:-8888}"
    ;;
  *)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
