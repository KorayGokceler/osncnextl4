"""
Driver for scripts/process_L4.py, with a live progress bar.

WHY A SEPARATE FILE: running the notebook rewrites its output cells, which
used to block every `git pull`.  With the logic here it stays versioned and
updates arrive through `git pull`, leaving only a few lines of call in the
notebook.

USAGE (notebook):

    from oscnext_l4.runner import configure_runner, run_process, run_all
    configure_runner(SAMPLES, PROCESS_PY, GCD)

    run_process("nue", n_frames=200)     # smoke test
    results = run_all(chunk_files=10)    # every sample, with a progress bar

process_L4.py prints machine-readable lines to stdout:

    [CHUNK] 3/10 files=30/100 booked=1840 elapsed=312.4
    [PROGRESS] frames=15000 physics=7100 booked=4300 elapsed=98.2 rate=152.7

The code here parses them and updates the bar.  With ipywidgets it is a real
widget, otherwise a single-line ASCII bar (which also works from a terminal).

IMPORTANT: the subprocess is started with `python -u`.  Without it Python
fully buffers stdout when writing into a pipe and no line arrives until the
job is over -- there would be no progress to show.
"""

import os
import re
import sys
import glob
import time
import shlex
import subprocess

try:
    from IPython.display import display
except ImportError:                       # notebook disinda da calissin
    def display(*a, **k):
        pass

try:
    import ipywidgets as _w
    _HAS_W = True
except ImportError:
    _HAS_W = False


# --- notebook'tan gelen tanimlar -------------------------------------------
_CFG = {"SAMPLES": None, "PROCESS_PY": None, "GCD": None}


def configure_runner(SAMPLES, PROCESS_PY, GCD):
    """
    Register the notebook's definitions with this module.  Called once.

    The paths are validated HERE.  Otherwise a missing process_L4.py makes the
    subprocess die with a meaningless "returncode=2" and no visible cause.
    The typical reason is a Jupyter server started from an old or deleted
    directory (the working directory then reads ~/.local/share/Trash/...),
    where a relative path is resolved against the wrong place.
    """
    _CFG["SAMPLES"] = SAMPLES
    _CFG["PROCESS_PY"] = PROCESS_PY
    _CFG["GCD"] = GCD

    cwd = os.getcwd()
    problems = []
    if not os.path.exists(PROCESS_PY):
        problems.append("process_L4.py not found: %s" % os.path.abspath(PROCESS_PY))
    if not os.path.exists(GCD):
        problems.append("GCD not found: %s" % GCD)
    if "Trash" in cwd or "/.Trash" in cwd:
        problems.append("the working directory is in the TRASH: %s" % cwd)

    if problems:
        print("[!] configure_runner: problems found")
        for x in problems:
            print("    " + x)
        print("    working directory: %s" % cwd)
        print()
        print("    Jupyter must be started from the RIGHT directory -- a kernel")
        print("    restart is not enough, the cwd comes from the server:")
        print("        cd ~/l4/osncnextl4 && python -m jupyter lab ...")
    else:
        print("configure_runner OK  (cwd: %s)" % cwd)


def _cfg(key):
    v = _CFG[key]
    if v is None:
        raise RuntimeError(
            "configure_runner(SAMPLES, PROCESS_PY, GCD) cagrilmamis -- "
            "notebook'ta bolum 1'deki hucreyi calistirin.")
    return v


_CHUNK_RE = re.compile(r"\[CHUNK\] (\d+)/(\d+) files=(\d+)/(\d+) booked=(\d+) elapsed=([\d.]+)")
_PROG_RE  = re.compile(r"\[PROGRESS\] frames=(\d+) physics=(\d+) booked=(\d+) elapsed=([\d.]+) rate=([\d.]+)")


def _fmt_eta(sec):
    if sec is None or sec != sec or sec < 0:
        return "?"
    sec = int(sec)
    # Units spelled out: s / min / h.  (Abbreviations like "8d32s" were
    # ambiguous -- d could read as minute or day, s as second or hour.)
    if sec < 60:
        return "%ds" % sec
    if sec < 3600:
        return "%ddk%02dsn" % (sec // 60, sec % 60)
    return "%dsa%02ddk" % (sec // 3600, (sec % 3600) // 60)


class _Bar:
    """A real widget when ipywidgets is available, else a one-line ASCII bar."""

    def __init__(self, label, total=None):
        self.label, self.total, self.t0 = label, total, time.time()
        if _HAS_W:
            self.w = _w.FloatProgress(value=0, min=0, max=(total or 1),
                                      description=label[:12],
                                      bar_style="info",
                                      layout=_w.Layout(width="45%"))
            self.txt = _w.HTML()
            self.box = _w.HBox([self.w, self.txt])
            display(self.box)

    def update(self, done, note=""):
        if self.total:
            frac = min(done / self.total, 1.0)
            eta = (time.time() - self.t0) * (1 - frac) / frac if frac > 0 else None
            body = "%d/%d  %.0f%%  ETA %s  %s" % (done, self.total, 100 * frac,
                                                  _fmt_eta(eta), note)
        else:
            frac, body = None, note
        if _HAS_W:
            if self.total:
                # total kurulumdan SONRA atanmis olabilir ([CHUNK] satirindan)
                # -> update the widget's max too, or the bar never fills.
                if self.w.max != self.total:
                    self.w.max = self.total
                self.w.value = min(done, self.total)
            else:
                self.w.max, self.w.value = 1, 0.5   # belirsiz
            self.txt.value = "<code>%s</code>" % body
        else:
            if self.total:
                n = int(30 * frac)
                bar = "#" * n + "-" * (30 - n)
                sys.stdout.write("\r  [%s] %s" % (bar, body))
            else:
                sys.stdout.write("\r  %s" % body)
            sys.stdout.flush()

    def done(self, note=""):
        if _HAS_W:
            self.w.bar_style = "success"
            if self.total:
                self.w.max = self.total
                self.w.value = self.total
            else:
                self.w.max, self.w.value = 1, 1
            self.txt.value = "<code>%s</code>" % note
        else:
            sys.stdout.write("\r  %s%s\n" % (note, " " * 30))
            sys.stdout.flush()

    def fail(self, note=""):
        if _HAS_W:
            self.w.bar_style = "danger"
            self.txt.value = "<code>%s</code>" % note
        else:
            sys.stdout.write("\r  [ERROR] %s\n" % note)
            sys.stdout.flush()


def l3_patterns(cfg):
    """
    The L3 input patterns of a sample, always as a list.

    A pass3 sample is one dataset, so `l3` was a single glob string.  A pass2
    sample can be several: NuE is 121122 AND 121291, NuMu is 141154 AND 141292.
    Both spellings are accepted so neither notebook has to change shape.
    """
    l3 = cfg["l3"]
    return list(l3) if isinstance(l3, (list, tuple)) else [l3]


def l3_files(cfg):
    """Every L3 file of a sample, de-duplicated, order preserved."""
    out = []
    for pat in l3_patterns(cfg):
        out.extend(sorted(glob.glob(pat)))
    return list(dict.fromkeys(out))


def run_process(name, n_frames=0, chunk_files=10, log_tail=15, bar=True,
                run_optional=False, extra_args=None):
    """
    Run process_L4.py for one sample and show live progress.

    chunk_files : how many L3 files go into one part (0 = a single part).
                  >0 ise gercek yuzde/ETA ve cokme sonrasi devam.
    n_frames    : >0 ise smoke test (chunk_files otomatik kapanir).
    extra_args  : process_L4.py'ye oldugu gibi eklenecek ek bayraklar,
                  orn. ["--micro-count-uncleaned"].
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"] if n_frames == 0 else cfg["hdf5"].replace(".hdf5", "_smoke.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if n_frames:
        chunk_files = 0                      # cannot be combined with --n

    cmd = ([sys.executable, "-u", _cfg("PROCESS_PY"),
            "--gcd", _cfg("GCD"), "--input"] + l3_patterns(cfg)
           + ["--output-hdf5", out] + cfg["flags"])
    if n_frames:
        cmd += ["--n", str(n_frames), "--scan", "off"]
    if chunk_files:
        cmd += ["--chunk-files", str(chunk_files)]
    if run_optional:
        cmd += ["--run-optional"]
    if extra_args:
        cmd += list(extra_args)

    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    b = _Bar(name) if bar else None
    tail, t0 = [], time.time()

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if len(tail) > 400:
                del tail[:200]

            m = _CHUNK_RE.match(line)
            if m and b:
                done, total, fdone, ftot, booked, el = m.groups()
                b.total = int(total)
                b.update(int(done), "files %s/%s  events %s" % (fdone, ftot, booked))
                continue

            m = _PROG_RE.match(line)
            if m and b:
                frames, phys, booked, el, rate = m.groups()
                if not b.total:
                    b.update(0, "frames %s  events %s  %s fr/s  %s" %
                             (frames, booked, rate, _fmt_eta(float(el))))
                continue

            if line.startswith(("Pre-scan", "  scanned:", "  Files to process",
                                "Parts:", "  [!]")) and b:
                b.update(0, line.strip()[:70])
    finally:
        p.wait()

    dt = time.time() - t0
    if p.returncode != 0:
        if b:
            b.fail("returncode=%d" % p.returncode)
        print("\n--- FAILED (returncode=%d) ---" % p.returncode)
        print("\n".join(tail[-40:]))
        return None

    parts = sorted(glob.glob(out.replace(".hdf5", "*.hdf5")))
    sz = sum(os.path.getsize(f) for f in parts) / 1e6
    msg = "%.1f MB, %d files, %.0f s" % (sz, len(parts), dt)
    if b:
        b.done(msg)
    if log_tail:
        print("\n".join(tail[-log_tail:]))
    print("-> %s  (%s)" % (out, msg))
    return out


def run_all(samples=None, chunk_files=10, jobs=1, run_optional=False,
            extra_args=None):
    """
    Process every sample in turn -- one bar each, plus an overall bar.

    samples       : sample names to process (default: all of SAMPLES)
    chunk_files   : how many L3 files go into one part.  >0 gives a real
                    yuzde/ETA ve cokme sonrasi kaldigi yerden devam.
    jobs          : SPEEDUP.  >1 processes each sample in N parallel workers
                    (run_process_parallel).  cobalt paylasilan makine:
                    8 is reasonable, 64 is not.
    run_optional : also compute the non-BDT variables (I3TensorOfInertia,
                    separation_in_cogs).  Neither is in Table 11/12.
    extra_args    : process_L4.py'ye oldugu gibi eklenecek ek bayraklar,
                    orn. ["--micro-count-uncleaned"] (pass2 karsilastirmasi).

    The default is jobs=1 -- the speedup is NOT automatic, it must be asked for.
    """
    names = list(samples or _cfg("SAMPLES"))
    overall = _Bar("TOPLAM", total=len(names))
    results = {}
    for i, name in enumerate(names):
        print("=" * 70)
        print(name)
        print("=" * 70)
        if jobs > 1:
            results[name] = run_process_parallel(
                name, jobs=jobs, chunk_files=chunk_files,
                run_optional=run_optional, extra_args=extra_args)
        else:
            results[name] = run_process(
                name, chunk_files=chunk_files, run_optional=run_optional,
                extra_args=extra_args)
        overall.update(i + 1, "%d/%d samples" % (i + 1, len(names)))
    ok = sum(v is not None for v in results.values())
    overall.done("%d/%d tamam" % (ok, len(names)))
    if ok < len(names):
        print("\n[!] Basarisiz: %s"
              % ", ".join(n for n, v in results.items() if v is None))
    return results


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
#
# The biggest speedup is here.  process_L4.py uses a single process on a
# single core, while cobalt has dozens.  Splitting the input files into N
# groups and running N separate process_L4.py workers is almost linear
# -- separate processes, separate output files, no shared state.
#
# Output names:  L4_nue_job0_part000.hdf5, L4_nue_job1_part000.hdf5, ...
# They all match the notebook's L4_nue*.hdf5 glob, and each part writes its
# own .meta.json, so n_l3_files sums correctly.
#
# CAREFUL: cobalt is a shared machine.  jobs=8 is fine, jobs=64 is not.

import threading


def _split(seq, n):
    """Split seq into n groups (the last groups may be one shorter)."""
    n = max(1, min(n, len(seq)))
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        m = k + (1 if j < r else 0)
        out.append(seq[i:i + m])
        i += m
    return [g for g in out if g]


def run_process_parallel(name, jobs=4, chunk_files=10, log_tail=10, bar=True,
                         run_optional=False, extra_args=None):
    """
    Bir ornegi N paralel surecte isle.

    jobs        : how many process_L4.py workers
    chunk_files : part size each worker uses internally, so a crash can be
                  resumed from where it stopped
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    files = l3_files(cfg)
    if not files:
        print("[!] %s: no L3 files -> %s"
              % (name, ", ".join(l3_patterns(cfg))))
        return None
    groups = _split(files, jobs)
    print("%s: %d L3 files -> %d workers (%s files/worker)"
          % (name, len(files), len(groups), "/".join(str(len(g)) for g in groups)))

    listdir = os.path.join(os.path.dirname(out), "_filelists")
    os.makedirs(listdir, exist_ok=True)

    procs, state = [], {}
    for j, grp in enumerate(groups):
        lst = os.path.join(listdir, "%s_job%d.txt" % (name, j))
        with open(lst, "w") as fh:
            fh.write("\n".join(grp) + "\n")
        base, ext = os.path.splitext(out)
        cmd = [sys.executable, "-u", _cfg("PROCESS_PY"),
               "--gcd", _cfg("GCD"),
               "--input-list", lst,
               "--scan", "off",          # scanned once, below
               "--output-hdf5", "%s_job%d%s" % (base, j, ext)] + cfg["flags"]
        if chunk_files:
            cmd += ["--chunk-files", str(chunk_files)]
        if run_optional:
            cmd += ["--run-optional"]
        if extra_args:
            cmd += list(extra_args)
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True,
                                      bufsize=1))
        state[j] = {"done": 0, "total": len(grp), "booked": 0, "tail": []}

    b = _Bar(name, total=len(files)) if bar else None
    lock = threading.Lock()

    def reader(j, p):
        for line in p.stdout:
            line = line.rstrip("\n")
            st = state[j]
            st["tail"].append(line)
            if len(st["tail"]) > 60:
                del st["tail"][:30]
            m = _CHUNK_RE.match(line)
            if m:
                _, _, fdone, _, booked, _ = m.groups()
                with lock:
                    st["done"] = int(fdone)
                    st["booked"] = int(booked)
                    if b:
                        d = sum(v["done"] for v in state.values())
                        bk = sum(v["booked"] for v in state.values())
                        b.update(d, "%d workers  events %d" % (len(procs), bk))
        p.wait()

    t0 = time.time()
    threads = [threading.Thread(target=reader, args=(j, p), daemon=True)
               for j, p in enumerate(procs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    dt = time.time() - t0
    rc = [p.returncode for p in procs]
    parts = sorted(glob.glob(out.replace(".hdf5", "*.hdf5")))
    sz = sum(os.path.getsize(f) for f in parts) / 1e6
    msg = "%.1f MB, %d files, %.0f s" % (sz, len(parts), dt)

    if any(r != 0 for r in rc):
        if b:
            b.fail("failed workers: %s" % [j for j, r in enumerate(rc) if r])
        for j, r in enumerate(rc):
            if r:
                print("\n--- job %d (rc=%d) ---" % (j, r))
                print("\n".join(state[j]["tail"][-log_tail:]))
        return None

    if b:
        b.done(msg)
    print("-> %s  (%s)" % (out, msg))
    return out
