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
import json
import shlex
import subprocess

try:
    from IPython.display import display
except ImportError:                       # so it also works outside a notebook
    def display(*a, **k):
        pass

try:
    import ipywidgets as _w
    _HAS_W = True
except ImportError:
    _HAS_W = False


# --- set from the notebook -------------------------------------------------
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
            "configure_runner(SAMPLES, PROCESS_PY, GCD) was not called -- "
            "run the cell in section 1 of the notebook.")
    return v


_CHUNK_RE = re.compile(r"\[CHUNK\] (\d+)/(\d+) files=(\d+)/(\d+) booked=(\d+) elapsed=([\d.]+)")
_PROG_RE  = re.compile(r"\[PROGRESS\] frames=(\d+) physics=(\d+) booked=(\d+) elapsed=([\d.]+) rate=([\d.]+)")


def _fmt_eta(sec):
    if sec is None or sec != sec or sec < 0:
        return "?"
    sec = int(sec)
    # Units spelled out in English, as all printed output is.  (The bare
    # abbreviations were ambiguous: "d" could read as minute or day, "s" as
    # second or hour -- and these three were in fact Turkish, dk/sn/sa, while
    # the comment above them claimed otherwise.)
    if sec < 60:
        return "%ds" % sec
    if sec < 3600:
        return "%dmin%02ds" % (sec // 60, sec % 60)
    return "%dh%02dmin" % (sec // 3600, (sec % 3600) // 60)


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
                # total may be set AFTER construction (from the [CHUNK] line)
                # -> update the widget's max too, or the bar never fills.
                if self.w.max != self.total:
                    self.w.max = self.total
                self.w.value = min(done, self.total)
            else:
                self.w.max, self.w.value = 1, 0.5   # indeterminate
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


def l3_files(cfg, max_files=0, fraction=0.0):
    """
    Every L3 file of a sample, de-duplicated, order preserved.

    max_files > 0 keeps only the first N of the WHOLE list.  The order is the
    sorted glob, so the same N files are taken every time -- a partial run is
    reproducible and can be extended later without reprocessing what is
    already there.

    fraction (0 < f <= 1) keeps a share of EACH PATTERN instead, and for a
    multi-pattern sample that is the only safe way to take a slice.

    WHY BOTH EXIST.  A sample whose `l3` is one glob is a single set and
    `max_files` slices it fine.  pass2's detector data is EIGHTEEN globs, one
    per run, ordered by year -- so `max_files` there would take all of 2012
    and 2013 and nothing of 2017.  That is precisely the seasonal dependence
    the note's run list was chosen to avoid: the atmospheric muon flux varies
    with the season, and a background drawn from one part of the calendar
    teaches the classifier that part rather than the muon.  `fraction` keeps
    every run represented in proportion.

    It takes a STRIDE (`files[::step]`), not the first half, because files
    within a run are time-ordered -- the first half of a run is the first half
    of its night, while a stride spans the whole of it.  Going from a fraction
    to the full set is still a superset, so a slice can be extended to
    everything without reprocessing; going from one fraction to another
    intermediate one is not, and would.
    """
    if fraction and not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1], not %r" % (fraction,))
    out = []
    for pat in l3_patterns(cfg):
        got = sorted(glob.glob(pat))
        if fraction and fraction < 1:
            step = max(1, int(round(1.0 / fraction)))
            got = got[::step]
        out.extend(got)
    out = list(dict.fromkeys(out))
    return out[:max_files] if max_files else out


def run_process(name, n_frames=0, chunk_files=10, log_tail=15, bar=True,
                run_optional=False, extra_args=None, max_files=0,
                fraction=0.0):
    """
    Run process_L4.py for one sample and show live progress.

    chunk_files : how many L3 files go into one part (0 = a single part).
                  >0 gives a real percentage/ETA and resumes after a crash.
    n_frames    : >0 means a smoke test (chunk_files is turned off).
    max_files   : >0 processes only the first N L3 files of the sample.  Some
                  pass2 sets are enormous (noise 888003 is 10000 files) and a
                  full run of one is hours of CPU and tens of GB; this is how
                  you take a slice.  The first N of the sorted list, so adding
                  more later does not reprocess what is done.
    fraction    : 0 < f <= 1 takes that share of EACH L3 pattern (see
                  l3_files).  For a multi-pattern sample -- pass2's detector
                  data is one pattern per run -- this is the slice to use;
                  max_files would take whole runs from the front of the list
                  and drop the later years entirely.
    extra_args  : extra flags passed to process_L4.py verbatim,
                  e.g. ["--micro-count-cleaned"].
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"] if n_frames == 0 else cfg["hdf5"].replace(".hdf5", "_smoke.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if n_frames:
        chunk_files = 0                      # cannot be combined with --n

    if max_files or fraction:
        files = l3_files(cfg, max_files, fraction)
        how = []
        if fraction:
            how.append("%.0f%% of each of the %d pattern(s)"
                       % (100 * fraction, len(l3_patterns(cfg))))
        if max_files:
            how.append("first %d overall" % max_files)
        print("  [i] %s: %d of %d L3 files (%s)"
              % (name, len(files), len(l3_files(cfg)), ", ".join(how)))
        inputs = ["--input"] + files
    else:
        inputs = ["--input"] + l3_patterns(cfg)

    cmd = ([sys.executable, "-u", _cfg("PROCESS_PY"),
            "--gcd", _cfg("GCD")] + inputs
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
            extra_args=None, max_files=None, fraction=None):
    """
    Process every sample in turn -- one bar each, plus an overall bar.

    samples       : sample names to process (default: all of SAMPLES)
    chunk_files   : how many L3 files go into one part.  >0 gives a real
                    percentage/ETA and resumes after a crash.
    jobs          : SPEEDUP.  >1 processes each sample in N parallel workers
                    (run_process_parallel).  cobalt is a SHARED machine:
                    8 is reasonable, 64 is not.
    run_optional  : also compute the non-BDT variables (I3TensorOfInertia,
                    separation_in_cogs).  Neither is in Table 11/12.
    max_files     : cap the number of L3 files per sample.  Either one number
                    for every sample, or a dict {sample: N} -- the pass2 sets
                    are wildly different sizes (nue 608, numu 1590, noise
                    10000), so per-sample is usually what you want:

                        run_all(jobs=8, max_files={"noise": 1000})

                    Samples not named in the dict are processed in full.
    fraction      : take a SHARE OF EACH L3 PATTERN instead of a count, as one
                    number or a dict {sample: f}.  Use this, not max_files,
                    for a sample whose `l3` is several patterns:

                        run_all(jobs=16, fraction={"data": 0.5})

                    pass2's detector data is one pattern per run, ordered by
                    year, so max_files there would take all of 2012-2013 and
                    none of 2017 -- the seasonal bias the note's run list was
                    chosen to avoid.  `fraction` halves every run instead, so
                    all six years stay represented in proportion.

    A sample with a per-run GCD (detector data) is routed to
    run_process_per_run automatically -- see the note at the dispatch.
    extra_args    : extra flags passed to process_L4.py verbatim,
                    e.g. ["--micro-count-cleaned"].

    The default is jobs=1 -- the speedup is NOT automatic, it must be asked for.
    """
    names = list(samples or _cfg("SAMPLES"))

    if isinstance(max_files, dict):
        unknown = [k for k in max_files if k not in _cfg("SAMPLES")]
        if unknown:
            raise ValueError("max_files names sample(s) that do not exist: %s"
                             % ", ".join(sorted(unknown)))
        caps = dict(max_files)
    else:
        caps = {n: max_files for n in names} if max_files else {}

    if isinstance(fraction, dict):
        unknown = [k for k in fraction if k not in _cfg("SAMPLES")]
        if unknown:
            raise ValueError("fraction names sample(s) that do not exist: %s"
                             % ", ".join(sorted(unknown)))
        fracs = dict(fraction)
    else:
        fracs = {n: fraction for n in names} if fraction else {}

    overall = _Bar("TOTAL", total=len(names))
    results = {}
    for i, name in enumerate(names):
        print("=" * 70)
        print(name)
        print("=" * 70)
        cap = caps.get(name, 0) or 0
        frac = fracs.get(name, 0.0) or 0.0
        # A sample that declares a per-pattern `gcd` LIST has one GCD per run
        # and cannot be processed in a single invocation -- detector data is
        # the case.  Dispatching on the SPEC rather than on the sample name
        # keeps the notebook production-agnostic: the same run_all() call
        # serves pass2 and pass3 and does not have to know that one of them
        # has a detector-data set and the other does not.
        if isinstance(_cfg("SAMPLES")[name].get("gcd"), (list, tuple)):
            results[name] = run_process_per_run(
                name, jobs=jobs, chunk_files=chunk_files,
                run_optional=run_optional, extra_args=extra_args,
                fraction=frac)
            overall.update(i + 1, "%d/%d samples" % (i + 1, len(names)))
            continue
        if jobs > 1:
            results[name] = run_process_parallel(
                name, jobs=jobs, chunk_files=chunk_files,
                run_optional=run_optional, extra_args=extra_args,
                max_files=cap, fraction=frac)
        else:
            results[name] = run_process(
                name, chunk_files=chunk_files, run_optional=run_optional,
                extra_args=extra_args, max_files=cap, fraction=frac)
        overall.update(i + 1, "%d/%d samples" % (i + 1, len(names)))
    ok = sum(v is not None for v in results.values())
    overall.done("%d/%d done" % (ok, len(names)))
    if ok < len(names):
        print("\n[!] Failed: %s"
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
                         run_optional=False, extra_args=None, max_files=0,
                fraction=0.0):
    """
    Process one sample in N parallel workers.

    jobs        : how many process_L4.py workers
    chunk_files : part size each worker uses internally, so a crash can be
                  resumed from where it stopped
    max_files   : >0 uses only the first N L3 files (see run_process)
    fraction    : 0 < f <= 1 takes that share of EACH pattern (see l3_files) --
                  the right knob for a sample with one pattern per run
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    files = l3_files(cfg, max_files, fraction)
    if max_files or fraction:
        print("  [i] %s: limited to the first %d of %d L3 files"
              % (name, len(files), len(l3_files(cfg))))
    if not files:
        print("[!] %s: no L3 files -> %s"
              % (name, ", ".join(l3_patterns(cfg))))
        return None
    groups = _split(files, jobs)
    print("%s: %d L3 files -> %d workers (%s files/worker)"
          % (name, len(files), len(groups), "/".join(str(len(g)) for g in groups)))

    listdir = os.path.join(os.path.dirname(out), "_filelists")
    os.makedirs(listdir, exist_ok=True)

    # CHANGING `jobs` BETWEEN RUNS IS SILENT CORRUPTION, so refuse it.
    #
    # Each worker writes L4_<name>_job<J>_part<C>.hdf5, and process_L4.py SKIPS
    # a part that already exists so a crashed run resumes.  But which files
    # worker J gets depends on `jobs`: at jobs=8 worker 1 starts at file 76, at
    # jobs=16 at file 38.  The path is the same either way, so the stale part
    # would be kept, the files it was supposed to hold would never be processed,
    # and its meta.json would describe the wrong chunk.  Nothing would warn.
    stale = []
    for j, grp in enumerate(groups):
        lst = os.path.join(listdir, "%s_job%d.txt" % (name, j))
        if not os.path.exists(lst):
            continue
        try:
            with open(lst) as fh:
                previous = [l.strip() for l in fh if l.strip()]
        except OSError:
            continue
        if previous == list(grp):
            continue
        base, ext = os.path.splitext(out)
        if glob.glob("%s_job%d*%s" % (base, j, ext)):
            stale.append(j)
    if stale:
        base, ext = os.path.splitext(out)
        # How much of this sample is already on disk.  Without it the message
        # reads as "something is broken", and the obvious response is the rm
        # it suggests -- which throws away a finished sample and hours of CPU.
        # Usually the sample is DONE and simply should not have been asked for
        # again, so say that first and name the way to leave it out.
        done_parts = sorted(glob.glob("%s_job*%s" % (base, ext)))
        done_files = 0
        for part in done_parts:
            try:
                with open(part + ".meta.json") as fh:
                    done_files += int(json.load(fh).get("n_l3_files") or 0)
            except Exception:
                pass
        raise RuntimeError(
            "%s: this sample was already processed with a DIFFERENT split "
            "(workers %s would now get different files), and output from that "
            "run is still there.\n"
            "Resuming would keep the old parts under the new worker numbering: "
            "some L3 files would never be processed, others would be counted "
            "under the wrong chunk, and no error would be raised.\n"
            "\n"
            "Already on disk: %d part(s) covering %d of this sample's %d L3 "
            "files.\n"
            "\n"
            "IF THAT SAMPLE IS FINISHED, do not reprocess it -- leave it out:\n"
            "    run_all(samples=[...without %r...], jobs=%d, ...)\n"
            "Otherwise either finish the run with the ORIGINAL `jobs`, or "
            "discard its output and start over:\n"
            "    rm -rf %s_job*%s %s_job*%s.meta.json %s\n"
            % (name, stale, len(done_parts), done_files, len(files),
               name, jobs, base, ext, base, ext, listdir))

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


# ---------------------------------------------------------------------------
# Detector data: one run at a time, because the GCD is per run
# ---------------------------------------------------------------------------

def run_gcd_pairs(cfg, fraction=0.0):
    """
    Pair each L3 pattern of a sample with its own GCD -> [(label, gcd, files)].

    Only a sample that declares a `gcd` LIST has per-run GCDs; everything else
    shares the production's single GCD and does not come through here.
    """
    pats = l3_patterns(cfg)
    gcds = cfg.get("gcd")
    if not isinstance(gcds, (list, tuple)) or len(gcds) != len(pats):
        raise ValueError(
            "this sample has no per-pattern `gcd` list (l3 has %d patterns, "
            "gcd is %r) -- use run_process/run_process_parallel instead"
            % (len(pats), gcds))

    out, missing = [], []
    for pat, gpat in zip(pats, gcds):
        files = sorted(glob.glob(pat))
        if fraction and fraction < 1:
            step = max(1, int(round(1.0 / fraction)))
            files = files[::step]
        found = sorted(glob.glob(gpat))
        # A run directory holds exactly one GCD.  Two would mean the glob is
        # catching something else, and silently taking the first would process
        # the run against the wrong detector state -- which is the whole reason
        # detector data cannot use the MC's averaged GCD.
        if len(found) != 1 or not files:
            missing.append((pat, gpat, len(files), len(found)))
            continue
        label = _run_label(pat)
        out.append((label, found[0], files))

    if missing:
        print("  [!] %d run(s) skipped:" % len(missing))
        for pat, gpat, nf, ng in missing:
            why = []
            if not nf:
                why.append("no L3 file")
            if ng != 1:
                why.append("%d GCD matches (expected 1)" % ng)
            print("      %s -- %s" % (_run_label(pat), ", ".join(why)))
            print("        l3 : %s" % pat)
            print("        gcd: %s" % gpat)
            # When the GCD glob matched nothing, the reason is almost always
            # the pattern rather than a missing file -- the 2015 runs carry
            # `_GCD.i3.gz` where the rest carry `.i3.zst`.  Printing what is
            # actually there turns that from a manual investigation into a
            # glance.
            if ng == 0:
                near = sorted(glob.glob(os.path.join(
                    os.path.dirname(gpat), "*GCD*")))
                if near:
                    print("        but the directory holds: %s"
                          % ", ".join(os.path.basename(f) for f in near[:3]))
                else:
                    print("        and nothing matching *GCD* is there at all")
    return out


def _run_label(pattern):
    """`Run00120200` out of a path, else a stable fallback."""
    m = re.search(r"(Run\d+)", pattern)
    return m.group(1) if m else re.sub(r"\W+", "_", pattern)[-24:]


def run_process_per_run(name, jobs=4, chunk_files=10, log_tail=10, bar=True,
                        run_optional=False, extra_args=None, fraction=0.0,
                        runs=None):
    """
    Process a sample ONE RUN AT A TIME, each against its own GCD.

    WHY THIS EXISTS.  Detector data cannot use the averaged MC GCD: the dead
    DOMs and the calibration are exactly what changes from run to run, and the
    L3 data files do not carry their own G/C/D frames (checked -- a pass2 L3
    data file holds only TrayInfo, DAQ and Physics).  process_L4.py takes a
    single --gcd, so the run is the unit of work.  That is also the shape the
    production ran in: run_oscNext.py takes ONE input file per invocation.

    The run is a better unit than a worker index anyway.  run_process_parallel
    has to refuse a changed `jobs` because its output paths are numbered by
    worker, so the same path means different files at a different split.  Here
    the output is named after the RUN, so `jobs` is free to change between
    invocations, an interrupted production resumes by simply running again,
    and one bad run can be redone on its own.

    It also makes the cross-check constraint hold for free: (Run, Event,
    SubEvent) is unique within one run's output, so pass2.match() can pair
    these files without the repeated-triple problem.

    runs        : process only these labels, e.g. ["Run00120200"].  Default
                  is every run the sample declares.
    jobs        : how many runs are processed CONCURRENTLY.
    fraction    : 0 < f <= 1 takes that share of each run's files.
    """
    import threading as _th

    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    base, ext = os.path.splitext(out)

    listdir = os.path.join(os.path.dirname(out), "_filelists")
    os.makedirs(listdir, exist_ok=True)

    pairs = run_gcd_pairs(cfg, fraction)
    if runs:
        want = set(runs)
        pairs = [p for p in pairs if p[0] in want]
    if not pairs:
        print("[!] %s: nothing to process" % name)
        return None

    # CHANGING `fraction` BETWEEN RUNS IS SILENT CORRUPTION, so refuse it.
    #
    # The run is a stable unit with respect to `jobs` -- that is why this path
    # needs no worker-split guard -- but it is NOT stable with respect to
    # `fraction`.  process_L4.py skips a part that already exists, and
    # `..._Run00120200_part000.hdf5` holds files [0, 2, 4, ...] after a
    # fraction=0.5 run and files [0..9] after a full one.  Same path, different
    # contents: the stale part would be kept, the files it should have held
    # would never be processed, and nothing would warn.
    #
    # The file list each run was produced from is already on disk, so the
    # check is a comparison rather than a heuristic.
    stale = []
    for label, _gcd, files in pairs:
        lst = os.path.join(listdir, "%s_%s.txt" % (name, label))
        if not os.path.exists(lst):
            continue
        try:
            with open(lst) as fh:
                previous = [l.strip() for l in fh if l.strip()]
        except OSError:
            continue
        if previous == list(files):
            continue
        if glob.glob("%s_%s*%s" % (base, label, ext)):
            stale.append((label, len(previous), len(files)))
    if stale:
        lines = "\n".join("      %-12s produced from %d files, would now get %d"
                           % (l, a, c) for l, a, c in stale)
        raise RuntimeError(
            "%s: %d run(s) were produced from a DIFFERENT file list, and that "
            "output is still there.\n%s\n"
            "\n"
            "process_L4.py skips a part that already exists, so resuming would "
            "keep parts whose contents no longer match their name: some L3 "
            "files would never be processed and no error would be raised.  "
            "This is what changing `fraction` between runs does.\n"
            "\n"
            "Either keep the fraction the run was produced with, or discard "
            "that output and redo it:\n"
            "    rm -rf %s_Run*%s %s_Run*%s.meta.json %s\n"
            % (name, len(stale), lines, base, ext, base, ext, listdir))

    n_files = sum(len(f) for _, _, f in pairs)
    print("%s: %d runs, %d L3 files, %d at a time" % (name, len(pairs),
                                                      n_files, jobs))
    for label, gcd, files in pairs:
        print("    %-12s %4d files   gcd %s" % (label, len(files),
                                                os.path.basename(gcd)))

    b = _Bar(name, total=n_files) if bar else None
    lock = _th.Lock()
    state = {}
    failed = []
    done_runs = [0]                 # a list so the reader closure can bump it
    t0 = time.time()

    def launch(label, gcd, files):
        lst = os.path.join(listdir, "%s_%s.txt" % (name, label))
        with open(lst, "w") as fh:
            fh.write("\n".join(files) + "\n")
        cmd = [sys.executable, "-u", _cfg("PROCESS_PY"),
               "--gcd", gcd,
               "--input-list", lst,
               "--scan", "off",
               "--output-hdf5", "%s_%s%s" % (base, label, ext)] + cfg["flags"]
        if chunk_files:
            cmd += ["--chunk-files", str(chunk_files)]
        if run_optional:
            cmd += ["--run-optional"]
        if extra_args:
            cmd += list(extra_args)
        state[label] = {"done": 0, "total": len(files), "booked": 0, "tail": []}
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)

    def reader(label, p):
        for line in p.stdout:
            line = line.rstrip("\n")
            st = state[label]
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
                        b.update(d, "%d/%d runs  events %d"
                                 % (done_runs[0], len(pairs), bk))
        p.wait()
        with lock:
            done_runs[0] += 1
            if p.returncode != 0:
                failed.append(label)

    # A pool rather than "launch them all": 18 runs at jobs=16 would otherwise
    # start 18 processes.  Slots free up as runs finish.
    #
    # Every thread is JOINED before the results are read.  Reaping on
    # `p.poll()` alone is not enough -- the reader is still draining the pipe
    # and still appending to `failed` for a moment after the process exits, so
    # reading `failed` at that point can miss a failure.
    started, queue = [], list(pairs)
    while queue or any(p.poll() is None for _, p, _ in started):
        while queue and sum(1 for _, p, _ in started
                            if p.poll() is None) < max(1, jobs):
            label, gcd, files = queue.pop(0)
            proc = launch(label, gcd, files)
            th = _th.Thread(target=reader, args=(label, proc), daemon=True)
            th.start()
            started.append((label, proc, th))
        time.sleep(1.0)              # a short sleep keeps this from spinning
    for _, _, th in started:
        th.join()

    dt = time.time() - t0
    parts = sorted(glob.glob("%s_Run*%s" % (base, ext)))
    sz = sum(os.path.getsize(f) for f in parts) / 1e6
    msg = "%.1f MB, %d files, %.0f s" % (sz, len(parts), dt)

    if failed:
        if b:
            b.fail("failed runs: %s" % sorted(failed))
        for label in sorted(failed):
            print("\n--- %s FAILED ---" % label)
            print("\n".join(state[label]["tail"][-log_tail:]))
        print("\nRe-run just those:  run_process_per_run(%r, runs=%s)"
              % (name, sorted(failed)))
        return None

    if b:
        b.done(msg)
    print("-> %s_Run*%s  (%s)" % (base, ext, msg))
    return out
