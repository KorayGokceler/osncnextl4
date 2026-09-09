"""
process_L4.py surucusu + canli ilerleme cubugu.

NEDEN AYRI DOSYA: notebook .gitignore'da (calistirilinca cikti hucreleri
degisiyor ve her `git pull`u blokluyordu).  Mantik burada durursa
versiyonlu kalir ve guncellemeler `git pull` ile gelir; notebook tarafinda
sadece birkac satirlik cagri kalir.

KULLANIM (notebook):

    from l4_run import configure_runner, run_process, run_all
    configure_runner(SAMPLES, PROCESS_PY, GCD)

    run_process("nue", n_frames=200)     # smoke test
    results = run_all(chunk_files=10)    # tum ornekler, ilerleme cubuguyla

process_L4.py stdout'a makine okunur satirlar basiyor:

    [CHUNK] 3/10 files=30/100 booked=1840 elapsed=312.4
    [PROGRESS] frames=15000 physics=7100 booked=4300 elapsed=98.2 rate=152.7

Buradaki kod onlari ayristirip cubugu gunceller.  ipywidgets varsa gercek
cubuk, yoksa tek satirlik ASCII cubuk (terminalden de calisir).

ONEMLI: alt surec `python -u` ile baslatiliyor.  Onsuz Python stdout'u bir
pipe'a yazarken tamamen tamponlar ve satirlar is bitene kadar gelmez --
ilerleme diye bir sey gorunmez.
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
    Notebook'taki tanimlari bu module tanit.  Bir kez cagrilir.

    Yollari BURADA dogruluyoruz.  Aksi halde process_L4.py bulunamadiginda
    alt surec anlamsiz bir "returncode=2" ile oluyor ve sebebi gorunmuyor.
    Tipik sebep: Jupyter eski/silinmis bir dizinden baslatilmis (calisma
    dizini ~/.local/share/Trash/... cikar) -- goreli "./process_L4.py"
    orada aranir.
    """
    _CFG["SAMPLES"] = SAMPLES
    _CFG["PROCESS_PY"] = PROCESS_PY
    _CFG["GCD"] = GCD

    cwd = os.getcwd()
    problems = []
    if not os.path.exists(PROCESS_PY):
        problems.append("process_L4.py bulunamadi: %s" % os.path.abspath(PROCESS_PY))
    if not os.path.exists(GCD):
        problems.append("GCD bulunamadi: %s" % GCD)
    if "Trash" in cwd or "/.Trash" in cwd:
        problems.append("calisma dizini COP KUTUSUNDA: %s" % cwd)

    if problems:
        print("[!] configure_runner: sorun var")
        for x in problems:
            print("    " + x)
        print("    calisma dizini: %s" % cwd)
        print()
        print("    Jupyter'i DOGRU dizinden baslatmis olmalisin -- kernel")
        print("    yeniden baslatmak yetmez, calisma dizini sunucudan gelir:")
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
    # Birimler acikca: sn / dk / sa.  ("8d32s" gibi kisaltmalar belirsizdi --
    # d hem dakika hem gun, s hem saniye hem saat okunabiliyordu.)
    if sec < 60:
        return "%dsn" % sec
    if sec < 3600:
        return "%ddk%02dsn" % (sec // 60, sec % 60)
    return "%dsa%02ddk" % (sec // 3600, (sec % 3600) // 60)


class _Bar:
    """ipywidgets varsa gercek cubuk, yoksa tek satir ASCII."""

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
                # -> widget'in max'ini da guncelle, yoksa cubuk hic dolmaz.
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
            sys.stdout.write("\r  [HATA] %s\n" % note)
            sys.stdout.flush()


def run_process(name, n_frames=0, chunk_files=10, log_tail=15, bar=True):
    """
    process_L4.py'yi bir ornek icin calistir, canli ilerleme goster.

    chunk_files : kac L3 dosyasi bir parcada islensin (0 = tek parca).
                  >0 ise gercek yuzde/ETA ve cokme sonrasi devam.
    n_frames    : >0 ise smoke test (chunk_files otomatik kapanir).
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"] if n_frames == 0 else cfg["hdf5"].replace(".hdf5", "_smoke.hdf5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if n_frames:
        chunk_files = 0                      # --n ile birlikte kullanilamaz

    cmd = [sys.executable, "-u", _cfg("PROCESS_PY"),
           "--gcd", _cfg("GCD"), "--input", cfg["l3"],
           "--output-hdf5", out] + cfg["flags"]
    if n_frames:
        cmd += ["--n", str(n_frames), "--scan", "off"]
    if chunk_files:
        cmd += ["--chunk-files", str(chunk_files)]

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
                b.update(int(done), "dosya %s/%s  olay %s" % (fdone, ftot, booked))
                continue

            m = _PROG_RE.match(line)
            if m and b:
                frames, phys, booked, el, rate = m.groups()
                if not b.total:
                    b.update(0, "frame %s  olay %s  %s fr/s  %s" %
                             (frames, booked, rate, _fmt_eta(float(el))))
                continue

            if line.startswith(("On tarama", "  taranan:", "  Islenecek dosya",
                                "Parca sayisi", "  [!]")) and b:
                b.update(0, line.strip()[:70])
    finally:
        p.wait()

    dt = time.time() - t0
    if p.returncode != 0:
        if b:
            b.fail("returncode=%d" % p.returncode)
        print("\n--- HATA (returncode=%d) ---" % p.returncode)
        print("\n".join(tail[-40:]))
        return None

    parts = sorted(glob.glob(out.replace(".hdf5", "*.hdf5")))
    sz = sum(os.path.getsize(f) for f in parts) / 1e6
    msg = "%.1f MB, %d dosya, %.0f s" % (sz, len(parts), dt)
    if b:
        b.done(msg)
    if log_tail:
        print("\n".join(tail[-log_tail:]))
    print("-> %s  (%s)" % (out, msg))
    return out


def run_all(samples=None, chunk_files=10):
    """
    Tum ornekleri sirayla isle -- her biri icin ayri cubuk + genel ilerleme.

    samples     : islenecek ornek adlari (varsayilan: SAMPLES'in hepsi)
    chunk_files : kac L3 dosyasi bir parcada islensin.  >0 ise gercek
                  yuzde/ETA ve cokme sonrasi kaldigi yerden devam.
    """
    names = list(samples or _cfg("SAMPLES"))
    overall = _Bar("TOPLAM", total=len(names))
    results = {}
    for i, name in enumerate(names):
        print("=" * 70)
        print(name)
        print("=" * 70)
        results[name] = run_process(name, chunk_files=chunk_files)
        overall.update(i + 1, "%d/%d ornek" % (i + 1, len(names)))
    ok = sum(v is not None for v in results.values())
    overall.done("%d/%d tamam" % (ok, len(names)))
    if ok < len(names):
        print("\n[!] Basarisiz: %s"
              % ", ".join(n for n, v in results.items() if v is None))
    return results


# ---------------------------------------------------------------------------
# Paralel calistirma
# ---------------------------------------------------------------------------
#
# En buyuk hizlanma burada.  process_L4.py tek surec ve tek cekirdek
# kullaniyor; cobalt'ta onlarca cekirdek var.  Girdi dosyalarini N gruba
# bolup N ayri process_L4.py surecinde islemek neredeyse dogrusal hizlanma
# verir -- ayri surecler, ayri cikti dosyalari, ortak durum yok.
#
# Cikti adlari:  L4_nue_job0_part000.hdf5, L4_nue_job1_part000.hdf5, ...
# Hepsi notebook'un L4_nue*.hdf5 glob'una uyar; her parca kendi
# .meta.json'ini yazar, n_l3_files dogru toplanir.
#
# DIKKAT: cobalt paylasilan bir makine.  jobs=8 makul, jobs=64 degil.

import threading


def _split(seq, n):
    """seq'i n gruba bol (son gruplar bir eksik olabilir)."""
    n = max(1, min(n, len(seq)))
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        m = k + (1 if j < r else 0)
        out.append(seq[i:i + m])
        i += m
    return [g for g in out if g]


def run_process_parallel(name, jobs=4, chunk_files=10, log_tail=10, bar=True):
    """
    Bir ornegi N paralel surecte isle.

    jobs        : kac process_L4.py sureci
    chunk_files : her surec kendi icinde kac dosyalik parcalar halinde islesin
                  (cokme sonrasi kaldigi yerden devam icin)
    """
    cfg = _cfg("SAMPLES")[name]
    out = cfg["hdf5"]
    os.makedirs(os.path.dirname(out), exist_ok=True)

    files = sorted(glob.glob(cfg["l3"]))
    if not files:
        print("[!] %s: L3 dosyasi yok -> %s" % (name, cfg["l3"]))
        return None
    groups = _split(files, jobs)
    print("%s: %d L3 dosyasi -> %d surec (%s dosya/surec)"
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
               "--scan", "off",          # tarama bir kez, asagida
               "--output-hdf5", "%s_job%d%s" % (base, j, ext)] + cfg["flags"]
        if chunk_files:
            cmd += ["--chunk-files", str(chunk_files)]
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
                        b.update(d, "%d surec  olay %d" % (len(procs), bk))
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
    msg = "%.1f MB, %d dosya, %.0f s" % (sz, len(parts), dt)

    if any(r != 0 for r in rc):
        if b:
            b.fail("basarisiz surec: %s" % [j for j, r in enumerate(rc) if r])
        for j, r in enumerate(rc):
            if r:
                print("\n--- job %d (rc=%d) ---" % (j, r))
                print("\n".join(state[j]["tail"][-log_tail:]))
        return None

    if b:
        b.done(msg)
    print("-> %s  (%s)" % (out, msg))
    return out
