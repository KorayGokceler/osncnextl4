"""
L4 HDF5 -> numpy: feature registry, yukleme ve agirliklar.

NEDEN AYRI DOSYA: notebook .gitignore'da; mantik burada durursa versiyonlu
kalir ve `git pull` ile guncellenir.

KULLANIM (notebook):

    from l4_data import (REGISTRY, AUX, NOISE_FEATURES, MUON_FEATURES,
                         WANTED, dump_tables, check_registry,
                         check_feature_map, load_sample, add_weights)

    TABLES = dump_tables(smoke_hdf5)
    check_registry(TABLES, NOISE_FEATURES)
    check_feature_map()

    data = {n: load_sample(n, SAMPLES, WANTED) for n in SAMPLES}
    data = {n: d for n, d in data.items() if d is not None}
    add_weights(data)

KRITIK SENKRONIZASYON: buradaki REGISTRY ile l4_classifier_module.py
icindeki FEATURE_MAP satir satir ayni olmali.  Biri digerinden farkli bir
kolon okursa model SESSIZCE yanlis tahmin uretir -- hata firlatmaz.
check_feature_map() bunu koddan kontrol eder.
"""

import os
import glob
import json

import numpy as np
import tables


IDX = ("Run", "Event", "SubEvent")


# BDT degisken adi -> (HDF5 tablosu, kolon)
L3V     = "IC2018_LE_L3_Vars"
HITSTAT = "SRTTWSplitInIcePulsesDCHitStatistics"
HITMULT = "SRTTWSplitInIcePulsesDCHitMultiplicity"

REGISTRY = {
    # --- noise BDT (Tablo 11) ---
    "NchCleaned":          (L3V, "NchCleaned"),
    "micro_count":         ("L4_micro_count", "STW_m3500p4000_DTW200"),
    # pass3 hdfwriter kolonu "lf_vel" (l4_classifier_module.FEATURE_MAP de
    # boyle); "LFVel" eski varsayimdi.  Teknik not Tablo 11 ise degiskeni
    # "L4_iLineFit.speed" (I3Particle alani) diye veriyor.  Ucu de ALTS'te.
    "iLineFit_speed":      ("L4_iLineFitParams", "lf_vel"),
    "fill_ratio":          ("L4_fill_ratio", "fill_ratio_from_mean"),
    "FullTimeLengthRatio": ("L4_FullTimeLengthRatio", "value"),

    # --- muon BDT (Tablo 12) ---
    "ICVetoHits":       (L3V, "ICVetoHits"),
    "RTVeto250Hits":    (L3V, "RTVeto250Hits"),
    "NAbove200Hits":    (L3V, "NAbove200Hits"),
    "VICH_nch":         ("L4_VICH_nch", "value"),
    "accumulated_time": ("L4_accumulated_time", "value"),
    "first_hlc_rho":    ("L4_first_hlc_rho", "value"),
    "cog_z":            (HITSTAT, "cog_z"),
    "z_sigma":          (HITSTAT, "z_sigma"),
    "z_travel":         (HITSTAT, "z_travel"),

    # --- aday / tarama icin ---
    "n_hit_doms":       (HITMULT, "n_hit_doms"),
    "C2HR6":            (L3V, "C2HR6"),
    "CausalVetoHits":   (L3V, "CausalVetoHits"),
    "VertexGuessZ":     (L3V, "VertexGuessZ"),
    "DCFiducialHits":   (L3V, "DCFiducialHits"),
}

NOISE_FEATURES = ["NchCleaned", "micro_count", "iLineFit_speed",
                  "fill_ratio", "FullTimeLengthRatio"]

MUON_FEATURES = ["ICVetoHits", "RTVeto250Hits", "NAbove200Hits", "VICH_nch",
                 "accumulated_time", "first_hlc_rho", "cog_z", "z_sigma",
                 "z_travel"]

# Agirlik hesabi icin gereken ek kolonlar (BDT girdisi DEGIL)
AUX = {
    "true_energy":   ("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
    "OneWeight":     ("I3MCWeightDict", "OneWeight"),
    "NEvents":       ("I3MCWeightDict", "NEvents"),
    "pdg":           ("I3MCWeightDict", "PrimaryNeutrinoType"),
    "n_flux_events": ("L4_n_flux_events", "value"),
    "noise_weight":  ("noise_weight", "value"),
}

def dump_tables(h5path, only=None, max_cols=40):
    """HDF5'teki tablolari ve kolonlarini listele."""
    if not os.path.exists(h5path):
        print("YOK:", h5path); return {}
    found = {}
    with tables.open_file(h5path, "r") as h5:
        for node in h5.walk_nodes("/", "Table"):
            cols = [c for c in node.colnames
                    if c not in ("Run", "Event", "SubEvent", "SubEventStream", "exists")]
            found[node.name] = (node.nrows, cols)
    for name in sorted(found):
        nrows, cols = found[name]
        if only and not any(o in name for o in only):
            continue
        print("%-45s %8d satir" % (name, nrows))
        for c in cols[:max_cols]:
            print("      %s" % c)
        if len(cols) > max_cols:
            print("      ... (+%d kolon)" % (len(cols) - max_cols))
    return found

def check_registry(found, names):
    """Registry'deki kolonlar HDF5'te gercekten var mi?"""
    ok, bad = [], []
    for n in names:
        tbl, col = REGISTRY.get(n, AUX.get(n, (None, None)))
        if tbl in found and col in found[tbl][1]:
            ok.append(n)
        else:
            bad.append((n, tbl, col))
    print("bulundu: %d/%d" % (len(ok), len(names)))
    for n, tbl, col in bad:
        why = "tablo yok" if tbl not in found else "kolon yok"
        print("  [!] %-22s %s[%s]  -- %s" % (n, tbl, col, why))
    return bad

def _ids(tbl):
    return (np.asarray(tbl.col("Run"), dtype=np.int64),
            np.asarray(tbl.col("Event"), dtype=np.int64),
            np.asarray(tbl.col("SubEvent"), dtype=np.int64))


# Isim varyasyonlari: meta-proje/pass surumune gore kolon adlari degisiyor.
# Dosyada GERCEKTEN hangisi varsa o kullanilir; hicbiri yoksa NaN + uyari.
ALTS = {
    "iLineFit_speed": [("L4_iLineFitParams", "lf_vel"),
                       ("L4_iLineFitParams", "LFVel"),
                       ("L4_iLineFit", "speed")],       # not Tablo 11 boyle diyor
    "micro_count":    [("L4_micro_count", "STW_m3500p4000_DTW200"),
                       ("L4_micro_count", "STW7500_DTW200")],
}

_warned_unresolved = set()


def resolve_one(name, available):
    """
    (tablo, kolon) ciftini dosyada GERCEKTEN var olana gore sec.

    available: {tablo_adi: set(kolon_adlari)}
    Bulunamazsa None.
    """
    cands = ALTS.get(name)
    if cands is None:
        pair = REGISTRY.get(name, AUX.get(name))
        cands = [pair] if pair else []
    for tbl, col in cands:
        if tbl in available and col in available[tbl]:
            return tbl, col
    return None


def load_one_file(path, wanted):
    """Tek HDF5 dosyasindan istenen degiskenleri oku -> {ad: dizi}."""
    with tables.open_file(path, "r") as h5:
        available = {n.name: set(n.colnames) for n in h5.walk_nodes("/", "Table")}

    need, unresolved = {}, []
    for name in wanted:
        hit = resolve_one(name, available)
        if hit:
            need.setdefault(hit[0], []).append((name, hit[1]))
        else:
            unresolved.append(name)
            # Cozulemeyen degisken -> NaN.  Ornek basina bir kez uyar.
            key = (os.path.basename(path).split("_part")[0], name)
            if key not in _warned_unresolved:
                _warned_unresolved.add(key)
                pair = REGISTRY.get(name, AUX.get(name))
                print("  [!] %-22s cozulemedi (%s) -> NaN" % (name, pair))

    with tables.open_file(path, "r") as h5:
        nodes = {n.name: n for n in h5.walk_nodes("/", "Table")}
        if "I3EventHeader" not in nodes:
            raise RuntimeError("%s: I3EventHeader tablosu yok" % path)

        ref = nodes["I3EventHeader"]
        run, ev, sub = _ids(ref)
        n = len(run)
        out = {"Run": run, "Event": ev, "SubEvent": sub}

        # Cozulemeyenler NaN olarak DOLDURULUR -- anahtar hic olmazsa
        # cagiran kod KeyError alir.
        for name in unresolved:
            out[name] = np.full(n, np.nan)
        ref_key = {k: i for i, k in enumerate(zip(run, ev, sub))}

        for tbl, cols in need.items():
            node = nodes.get(tbl)
            if node is None:
                for name, _ in cols:
                    out[name] = np.full(n, np.nan)
                continue

            r2, e2, s2 = _ids(node)
            same = (len(r2) == n and np.array_equal(r2, run)
                    and np.array_equal(e2, ev) and np.array_equal(s2, sub))
            if same:
                idx = None                       # hizali: dogrudan kullan
            else:
                idx = np.full(n, -1, dtype=np.int64)
                for j, k in enumerate(zip(r2, e2, s2)):
                    i = ref_key.get(k)
                    if i is not None:
                        idx[i] = j

            for name, col in cols:
                if col not in node.colnames:
                    out[name] = np.full(n, np.nan)
                    continue
                v = np.asarray(node.col(col), dtype=np.float64)
                if idx is None:
                    out[name] = v
                else:
                    a = np.full(n, np.nan)
                    ok = idx >= 0
                    a[ok] = v[idx[ok]]
                    out[name] = a
    return out


def n_l3_files(h5path):
    """
    Bu HDF5'in KAC L3 dosyasindan uretildigini oku.

    KRITIK: agirlik normalizasyonu (OneWeight/n_flux/n_files) L3 DOSYA
    sayisina bolunmeli.  Onceki kod HDF5 dosya sayisini kullaniyordu --
    100 L3 dosyasi tek HDF5'e book edildiginde bolen 1 cikiyordu, yani
    agirliklar 100 KAT buyuktu.  process_L4.py artik her ciktinin yanina
    <cikti>.meta.json yaziyor; dogru sayi orada.
    """
    meta = h5path + ".meta.json"
    if os.path.exists(meta):
        try:
            with open(meta) as fh:
                return int(json.load(fh)["n_l3_files"])
        except (OSError, ValueError, KeyError):
            pass
    return None


def load_sample(name, SAMPLES, wanted, max_files=None):
    """Bir ornegin tum HDF5 dosyalarini oku ve birlestir."""
    pattern = SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")
    files = sorted(f for f in glob.glob(pattern) if "_smoke" not in f)
    if not files:
        files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    if not files:
        print("[!] %s: HDF5 bulunamadi (%s)" % (name, pattern)); return None

    parts = [load_one_file(f, wanted) for f in files]
    keys = parts[0].keys()
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}

    # --- L3 dosya sayisi: agirliklarin boleni ---
    per_file = [n_l3_files(f) for f in files]
    if all(v is not None for v in per_file):
        n_l3 = sum(per_file)
        src = "meta.json"
    else:
        n_l3 = SAMPLES[name].get("n_l3_files") or len(files)
        src = "SAMPLES['%s']['n_l3_files'] (meta.json yok!)" % name
        print("  [!] %s: bazi HDF5'lerin meta.json'i yok -> n_l3_files=%d (%s)"
              % (name, n_l3, src))
        print("      Agirliklar YANLIS olabilir.  process_L4.py'yi bu surumle")
        print("      yeniden calistirin ya da n_l3_files'i elle dogrulayin.")
    data["_n_files"] = float(n_l3)
    data["_n_hdf5"] = len(files)

    print("%-8s %8d olay, %d HDF5, %d L3 dosyasi (%s)"
          % (name, len(data["Run"]), len(files), n_l3, src))
    return data


WANTED = sorted(set(NOISE_FEATURES) | set(MUON_FEATURES) | set(AUX))


# ---------------------------------------------------------------------------
# Agirliklar
# ---------------------------------------------------------------------------
#
# Bolen d["_n_files"] = L3 DOSYA sayisi.  HDF5 dosya sayisi DEGIL --
# load_sample bunu <cikti>.hdf5.meta.json icindeki n_l3_files'tan topluyor.
# (Eski kod HDF5 sayisini kullaniyordu: 100 L3 dosyasi tek HDF5'e book
# edilince bolen 1 cikiyor, agirliklar 100 KAT buyuk oluyordu.)

NORM, GAMMA = 2e-2, -3.0
NU_FRAC, NUBAR_FRAC = 0.7, 0.3

# Vuvuzela noise_weight birimi: pass3 -> 1/ns (x1e9), pass2 -> zaten Hz
NOISE_NS_SCALE = 1e9


def genie_weight(d):
    """w [Hz] = OneWeight * flux(E) / n_flux / n_files,  flux = NORM * E^GAMMA."""
    E = d["true_energy"]
    ow = d["OneWeight"]
    flux = NORM * np.power(E, GAMMA, where=E > 0, out=np.full_like(E, np.nan))

    n_flux = d.get("n_flux_events", np.full_like(E, np.nan)).copy()
    missing = ~np.isfinite(n_flux)
    if missing.any():
        frac = np.where(d["pdg"] < 0, NUBAR_FRAC, NU_FRAC)
        n_flux[missing] = (d["NEvents"] * frac)[missing]
        print("  [i] %d olayda n_flux_events yok -> NEvents * (%.1f/%.1f)"
              % (missing.sum(), NU_FRAC, NUBAR_FRAC))
        print("      (--genie bayragi unutulmus olabilir)")
    return ow * flux / n_flux / d["_n_files"]


def noise_weight(d):
    return d["noise_weight"] * NOISE_NS_SCALE / d["_n_files"]


def corsika_weight(d):
    """simweights yoksa kaba yaklasim -- mutlak oran guvenilmez."""
    print("  [!] CORSIKA agirligi yaklasik (simweights entegrasyonu yok). "
          "Mutlak oranlara guvenmeyin; sekil/egitim icin yeterli.")
    return np.ones(len(d["Run"])) / d["_n_files"]


WEIGHTERS = {"nue": genie_weight, "numu": genie_weight,
             "noise": noise_weight, "corsika": corsika_weight}

# Teknik not Tablo 13, L3 oranlari [mHz] -- mertebe kontrolu icin
L3_RATES_MHZ = {"nue": 0.95, "numu": 3.77, "corsika": 505.0, "noise": 36.6}


def add_weights(data, weighters=None):
    """
    Her ornege w_phys [Hz] ekle ve Tablo 13 ile mertebe karsilastirmasi bas.

    maks/toplam > %5 ise tek bir olay orani domine ediyor demektir:
    istatistik yetersiz ya da agirlik hesabinda sorun var.
    """
    weighters = weighters or WEIGHTERS
    for name, d in data.items():
        print(name)
        fn = weighters.get(name)
        if fn is None:
            print("  [!] agirlik fonksiyonu yok -> w_phys = NaN")
            d["w_phys"] = np.full(len(d["Run"]), np.nan)
            continue
        w = np.asarray(fn(d), dtype=np.float64)
        w[~np.isfinite(w)] = 0.0
        d["w_phys"] = w

        tot = w.sum()
        frac = 100 * w.max() / tot if tot > 0 else np.nan
        print("  toplam oran = %.4e Hz  (%.3f mHz),  maks/toplam = %.2f%%"
              % (tot, 1e3 * tot, frac))
        if frac > 5:
            print("      [!] tek olay orani domine ediyor (>%5) -- istatistik "
                  "yetersiz ya da agirlik hatali")

        ref = L3_RATES_MHZ.get(name)
        if ref:
            ratio = (1e3 * tot) / ref if ref else np.nan
            flag = "" if 0.1 <= ratio <= 10 else "   [!] MERTEBE SAPMASI"
            print("      Tablo 13 (L3, pass2): %.3f mHz  ->  bizim/nota oran "
                  "= %.2f%s" % (ref, ratio, flag))
    return data


# ---------------------------------------------------------------------------
# REGISTRY <-> FEATURE_MAP tutarliligi
# ---------------------------------------------------------------------------

def read_feature_map(path=None):
    """
    l4_classifier_module.py icindeki FEATURE_MAP'i **AST ile** oku.

    Import etmiyoruz cunku o modul icetray'e bagimli; boylece bu kontrol
    duz bir python kernel'inde de calisir.  (Repoda ayni yontem eski
    notebook'un anahtar okumasinda da kullaniliyor.)
    """
    import ast
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "l4_classifier_module.py")
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path).read())

    ns = {}          # modul seviyesindeki string sabitleri (L3V, HITSTAT, ...)
    fmap = None

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return ns[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(ev(e) for e in node.elts)
        raise ValueError

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        if tgt.id == "FEATURE_MAP" and isinstance(node.value, ast.Dict):
            fmap = {}
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    fmap[ev(k)] = tuple(ev(v))
                except (ValueError, KeyError, TypeError):
                    pass
        else:
            try:
                ns[tgt.id] = ev(node.value)
            except (ValueError, KeyError, TypeError):
                pass
    return fmap


def check_feature_map(verbose=True):
    """
    l4_classifier_module.FEATURE_MAP ile REGISTRY ayni mi?

    Egitimde bir kolon, frame'e uygularken baska bir kolon okunursa model
    sessizce sacmalar -- hata firlatmaz.  Bu yuzden koddan kontrol ediyoruz.
    Uyusmayanlarin listesini dondurur (bos liste = uyumlu, None = okunamadi).
    """
    fmap = read_feature_map()
    if not fmap:
        if verbose:
            print("[!] FEATURE_MAP okunamadi (l4_classifier_module.py yok "
                  "ya da bicimi degismis)")
        return None

    # TEHLIKELI: ayni isim, FARKLI kolon -> egitim ve uygulama baska seyi okur
    conflict = []
    # Zararsiz: FEATURE_MAP'te olup REGISTRY'de olmayanlar.  FEATURE_MAP bir
    # ust kume (aday degiskenler de icinde), bu beklenen bir durum.
    only_fmap = []

    for name, pair in sorted(fmap.items()):
        mine = REGISTRY.get(name)
        if mine is None:
            only_fmap.append(name)
        elif tuple(mine) != tuple(pair):
            conflict.append((name, tuple(mine), pair))

    for name in sorted(set(NOISE_FEATURES) | set(MUON_FEATURES)):
        if name in REGISTRY and name not in fmap:
            conflict.append((name, tuple(REGISTRY[name]), "FEATURE_MAP'te YOK"))

    if verbose:
        if conflict:
            print("[!] CAKISMA -- ayni degisken, farkli kolon.")
            print("    Egitimde biri, frame'e uygularken digeri okunur; model")
            print("    hata firlatmadan sacmalar:")
            for name, a, b in conflict:
                print("    %-22s REGISTRY=%s  FEATURE_MAP=%s" % (name, a, b))
        else:
            print("REGISTRY <-> FEATURE_MAP cakismasi YOK "
                  "(%d ortak degisken)" % (len(set(fmap) & set(REGISTRY))))
        if only_fmap:
            print("  (bilgi) FEATURE_MAP'te fazladan %d aday degisken var: %s"
                  % (len(only_fmap), ", ".join(only_fmap[:6])
                     + (" ..." if len(only_fmap) > 6 else "")))
    return conflict
