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

# Tablo 12 SIRASIYLA 10 degisken listeliyor.  NchCleaned hem noise hem muon
# BDT'sinde girdi -- ilk yazimda muon listesinden atlanmisti.
MUON_FEATURES = ["ICVetoHits", "RTVeto250Hits", "NchCleaned", "NAbove200Hits",
                 "VICH_nch",
                 "accumulated_time", "first_hlc_rho", "cog_z", "z_sigma",
                 "z_travel"]

# Agirlik hesabi icin gereken ek kolonlar (BDT girdisi DEGIL)
#
# DIKKAT: bunlarin hepsi HER ORNEKTE YOK.  noise_weight sadece vuvuzela
# gurultu MC'sinde, OneWeight/PrimaryNeutrino* sadece GENIE'de bulunur.
# Hepsini tek bir ornege karsi kontrol etmek YANLIS ALARM uretir --
# aux_for(kind) ile ilgili olanlari secin.
AUX = {
    "true_energy":   ("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
    "OneWeight":     ("I3MCWeightDict", "OneWeight"),
    "NEvents":       ("I3MCWeightDict", "NEvents"),
    "pdg":           ("I3MCWeightDict", "PrimaryNeutrinoType"),
    "n_flux_events": ("L4_n_flux_events", "value"),
    # Eski (LightGBM) notebook'ta dogrulanmis hali: kolon "weight".
    # "value" yanlis varsayimdi -> gurultu agirligi NaN kaliyordu.  ALTS'te ikisi de var.
    "noise_weight":  ("noise_weight", "weight"),
    # CORSIKA -- simweights yoksa elle hesap icin
    "cwm_Weight":       ("CorsikaWeightMap", "Weight"),
    "cwm_NEvents":      ("CorsikaWeightMap", "NEvents"),
    "cwm_OverSampling": ("CorsikaWeightMap", "OverSampling"),
}

# AUX kolonu -> hangi ornek turlerinde bulunur (SAMPLES[...]["kind"])
AUX_KINDS = {
    "true_energy":   ("signal",),
    "OneWeight":     ("signal",),
    "NEvents":       ("signal",),
    "pdg":           ("signal",),
    "n_flux_events": ("signal",),
    "noise_weight":  ("noise_bg",),
    "cwm_Weight":       ("muon_bg",),
    "cwm_NEvents":      ("muon_bg",),
    "cwm_OverSampling": ("muon_bg",),
}


def aux_for(kind):
    """Bu ornek turunde BEKLENEN AUX kolonlari."""
    return [k for k, kinds in AUX_KINDS.items() if kind in kinds]


def _table_nodes(h5):
    """
    Dosyadaki GERCEK veri tablolari -> {ad: node}.

    NEDEN AYRI FONKSIYON: h5.walk_nodes("/", "Table") ALT GRUPLARA DA iner.
    hdfwriter her anahtar icin iki tablo yazar:

        /L4_VICH_nch                 <- gercek veri  (Run, Event, ..., value)
        /__I3Index__/L4_VICH_nch     <- index        (exists, start, stop)

    Sozlugu leaf isimle kurunca ikisi CAKISIYOR ve index tablosu gercek
    veriyi eziyordu; sonuc: her tablo "start/stop" kolonlu gorunuyor,
    aranan kolonlar bulunamiyor, TUM DEGISKENLER NaN oluyordu.

    Cozum: __I3Index__ altini atla, ayni isim iki yerde varsa KOKE EN YAKIN
    olani sec.
    """
    best = {}
    for node in h5.walk_nodes("/", "Table"):
        path = node._v_pathname
        if "__I3Index__" in path:
            continue
        depth = path.count("/")
        if node.name not in best or depth < best[node.name][0]:
            best[node.name] = (depth, node)
    return {k: v[1] for k, v in best.items()}


def _index_node(h5, name):
    """
    hdfwriter'in /__I3Index__/<anahtar> tablosu -- yoksa None.

    Bu tablo her FRAME icin bir satir tutar: exists (anahtar bu frame'de
    var mi), start/stop (veri tablosundaki satir araligi).  Eslestirmenin
    DOGRU yolu budur; Run/Event/SubEvent uzerinden eslestirme MC'de
    guvenilmez cunku ucluler bir parca icinde tekrarlayabiliyor
    (run_id = set no, event_id her L3 dosyasinda sifirdan basliyor).
    """
    try:
        return h5.get_node("/__I3Index__/" + name)
    except Exception:
        return None


def dump_tables(h5path, only=None, max_cols=40):
    """HDF5'teki tablolari ve kolonlarini listele."""
    if not os.path.exists(h5path):
        print("YOK:", h5path); return {}
    found = {}
    with tables.open_file(h5path, "r") as h5:
        for node in _table_nodes(h5).values():
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

def check_registry(found, names, verbose=True):
    """
    Registry'deki kolonlar HDF5'te gercekten var mi?

    ALTS'i de dener -- yukleme sirasinda hangi varyant kullanilacaksa
    burada da AYNISI raporlanir.  (Onceden sadece REGISTRY'ye bakiyordu,
    yani ALTS bir varyanti cozse bile "kolon yok" diyordu.)

    found: dump_tables ciktisi  {tablo: (nrows, [kolonlar])}
    Bulunamayanlarin listesini dondurur.
    """
    available = {t: set(cols) for t, (_, cols) in found.items()}
    ok, bad, alt_used = [], [], []
    for n in names:
        hit = resolve_one(n, available)
        if hit is None:
            bad.append((n, REGISTRY.get(n, AUX.get(n))))
            continue
        ok.append(n)
        first = (ALTS.get(n) or [REGISTRY.get(n, AUX.get(n))])[0]
        if tuple(hit) != tuple(first):
            alt_used.append((n, hit, first))

    if verbose:
        print("bulundu: %d/%d" % (len(ok), len(names)))
        for n, hit, first in alt_used:
            print("  [i] %-22s %s[%s]  (ilk aday %s[%s] yoktu)"
                  % (n, hit[0], hit[1], first[0], first[1]))
        for n, pair in bad:
            tbl = pair[0] if pair else "?"
            why = "tablo yok" if tbl not in available else "kolon yok"
            print("  [!] %-22s %s  -- %s" % (n, pair, why))
            if tbl in available:
                cols = sorted(available[tbl])
                print("      %s icindeki kolonlar: %s"
                      % (tbl, ", ".join(cols[:12]) + (" ..." if len(cols) > 12 else "")))
    return bad


def _ids(tbl):
    return (np.asarray(tbl.col("Run"), dtype=np.int64),
            np.asarray(tbl.col("Event"), dtype=np.int64),
            np.asarray(tbl.col("SubEvent"), dtype=np.int64))


def _has_duplicate_ids(r, e, s):
    """Ucluler benzersiz mi?  (hizli kontrol)"""
    if len(r) == 0:
        return False
    arr = np.empty(len(r), dtype=[("r", np.int64), ("e", np.int64), ("s", np.int64)])
    arr["r"], arr["e"], arr["s"] = r, e, s
    return len(np.unique(arr)) < len(arr)


# Isim varyasyonlari: meta-proje/pass surumune gore kolon adlari degisiyor.
# Dosyada GERCEKTEN hangisi varsa o kullanilir; hicbiri yoksa NaN + uyari.
ALTS = {
    "iLineFit_speed": [("L4_iLineFitParams", "lf_vel"),
                       ("L4_iLineFitParams", "LFVel"),
                       ("L4_iLineFit", "speed")],       # not Tablo 11 boyle diyor
    "micro_count":    [("L4_micro_count", "STW_m3500p4000_DTW200"),
                       ("L4_micro_count", "STW7500_DTW200")],
    # Tablo 11: "L4_fill_ratio.fill_ratio_from_mean".  hdfwriter'in
    # I3FillRatioInfo cevirici isimlendirmesi surume gore degisiyor.
    # ASAGIDAKILERIN HEPSI AYNI BUYUKLUK (mean'e gore fill ratio) --
    # from_rms / from_nch gibi FARKLI buyuklukleri BILEREK koymuyoruz,
    # yoksa sessizce baska bir fizik okunur.
    "noise_weight":   [("noise_weight", "weight"),
                       ("noise_weight", "value")],
    "fill_ratio":     [("L4_fill_ratio", "fill_ratio_from_mean"),
                       ("L4_fill_ratio", "fillratio_from_mean"),
                       ("L4_fill_ratio", "fillRatioFromMean"),
                       ("L4_fill_ratio", "FillRatioFromMean")],
}

_warned_unresolved = set()


def _sample_tag(path):
    """L4_nue_job3_part002.hdf5 -> 'L4_nue'  (uyarilari ornek basina teklestirmek icin)"""
    b = os.path.basename(path)
    for sep in ("_job", "_part"):
        if sep in b:
            b = b.split(sep)[0]
    return b


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
        available = {k: set(n.colnames) for k, n in _table_nodes(h5).items()}

    need, unresolved = {}, []
    for name in wanted:
        hit = resolve_one(name, available)
        if hit:
            need.setdefault(hit[0], []).append((name, hit[1]))
        else:
            unresolved.append(name)
            # Cozulemeyen degisken -> NaN.  ORNEK basina bir kez uyar
            # (dosya basina degil: 16 parcada 16 kez basiyordu).
            key = (_sample_tag(path), name)
            if key not in _warned_unresolved:
                _warned_unresolved.add(key)
                pair = REGISTRY.get(name, AUX.get(name))
                tbl = pair[0] if pair else "?"
                if tbl not in available:
                    print("  [!] %-20s TABLO YOK: %s -> NaN" % (name, tbl))
                else:
                    cols = sorted(available[tbl])
                    print("  [!] %-20s %s icinde '%s' kolonu YOK -> NaN"
                          % (name, tbl, pair[1]))
                    print("      mevcut kolonlar: %s"
                          % (", ".join(cols[:10]) + (" ..." if len(cols) > 10 else "")))

    with tables.open_file(path, "r") as h5:
        nodes = _table_nodes(h5)
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

        # Tekrar eden Run/Event/SubEvent var mi?  (bkz. _occurrence_keys)
        dup = _has_duplicate_ids(run, ev, sub)
        if dup:
            key = (_sample_tag(path), "__dup__")
            if key not in _warned_unresolved:
                _warned_unresolved.add(key)
                print("  (i) %s: Run/Event/SubEvent ucluleri parca icinde "
                      "tekrarliyor (normal: bir parcada birden fazla L3 "
                      "dosyasi var) -- eslestirme __I3Index__ uzerinden"
                      % _sample_tag(path))
        ref_key = None          # gerektiginde kurulur (pahali)

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
                idx = None
                # 1) DOGRU YOL: hdfwriter'in frame indeksi
                inode = _index_node(h5, tbl)
                if inode is not None and len(inode) == n \
                        and "exists" in inode.colnames and "start" in inode.colnames:
                    ex = np.asarray(inode.col("exists")).astype(bool)
                    st = np.asarray(inode.col("start"), dtype=np.int64)
                    idx = np.where(ex, st, -1)
                    idx[idx >= len(r2)] = -1     # savunma
                elif dup:
                    # 2) Index yok VE ucluler tekrarliyor -> guvenilir
                    #    eslestirme MUMKUN DEGIL.  Sessizce yanlis
                    #    eslestirmektense NaN birak ve SOYLE.
                    key = (_sample_tag(path), "__ambig__" + tbl)
                    if key not in _warned_unresolved:
                        _warned_unresolved.add(key)
                        print("  [!] %s: __I3Index__ yok ve Run/Event/SubEvent "
                              "tekrarliyor -> guvenilir eslestirme yok, NaN"
                              % tbl)
                    idx = np.full(n, -1, dtype=np.int64)
                else:
                    # 3) Index yok ama ucluler benzersiz -> sozlukle esle
                    if ref_key is None:
                        ref_key = {k: i for i, k in
                                   enumerate(zip(run.tolist(), ev.tolist(),
                                                 sub.tolist()))}
                    idx = np.full(n, -1, dtype=np.int64)
                    for j, k in enumerate(zip(r2.tolist(), e2.tolist(),
                                              s2.tolist())):
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


def sample_files(name, SAMPLES, include_smoke=False):
    """
    Bir ornegin VAR OLAN HDF5 dosyalarini bul.

    L4_nue.hdf5, L4_nue_part000.hdf5, ... hepsi eslesir.  _smoke dosyalari
    varsayilan olarak DISLANIR (kucuk test ciktisi, uretim degil); hic
    uretim dosyasi yoksa onlara duser.
    """
    pattern = SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")
    files = sorted(glob.glob(pattern))
    if not include_smoke:
        real = [f for f in files if "_smoke" not in f]
        if real:
            return real
    return files


def find_hdf5(name, SAMPLES, verbose=True):
    """
    Ornekten ILK var olan HDF5'i dondur -- dump_tables icin.

    Sabit bir "_smoke.hdf5" yolu yazmak yerine bunu kullanin: smoke test
    calistirilmadiysa o dosya YOKTUR ama uretim ciktisi vardir.
    """
    files = sample_files(name, SAMPLES)
    if files:
        if verbose:
            print("%s: %d HDF5 bulundu, ilki inceleniyor -> %s"
                  % (name, len(files), os.path.basename(files[0])))
        return files[0]
    if verbose:
        print("[!] %s icin HDF5 yok: %s"
              % (name, SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")))
        print("    Once bolum 1'i calistirin (run_all).")
    return None


def load_sample(name, SAMPLES, wanted, max_files=None):
    """Bir ornegin tum HDF5 dosyalarini oku ve birlestir."""
    # Bu ornekte BULUNMASI BEKLENMEYEN AUX kolonlarini isteme.  Aksi halde
    # her CORSIKA dosyasi icin "OneWeight yok" gibi yaniltici uyarilar cikar.
    kind = SAMPLES[name].get("kind")
    if kind:
        drop = set(AUX) - set(aux_for(kind))
        skipped = [w for w in wanted if w in drop]
        wanted = [w for w in wanted if w not in drop]
        if skipped:
            print("  (%s: bu ornekte beklenmeyen %d AUX kolonu atlandi: %s)"
                  % (name, len(skipped), ", ".join(sorted(skipped))))

    files = sample_files(name, SAMPLES)
    if max_files:
        files = files[:max_files]
    if not files:
        print("[!] %s: HDF5 bulunamadi (%s)"
              % (name, SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")))
        return None

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
    # CORSIKA agirligi dosyalari YENIDEN acmak zorunda (simweights HDF5'i
    # dogrudan okuyor), o yuzden listeyi ve parca basina L3 sayisini sakla.
    data["_files"] = list(files)
    data["_n_l3_per_file"] = [n_l3_files(f) for f in files]

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


def _corsika_weight_manual(d):
    """
    simweights yoksa: CorsikaWeightMap.Weight / (NEvents * OverSampling).

    Eski (LightGBM) notebook'un fallback'i ile ayni.  Mutlak oran
    guvenilmez ama en azindan olaylar SPEKTRUMA gore agirliklanir --
    duz 1.0 vermekten cok daha iyisi.
    """
    print("  [!] simweights YOK -> CorsikaWeightMap ile yaklasik agirlik.")
    print("      Mutlak oran guvenilmez; egitim icin spektrum en azindan dogru.")
    w = np.asarray(d.get("cwm_Weight"), dtype=np.float64)
    nev = np.asarray(d.get("cwm_NEvents"), dtype=np.float64)
    osamp = np.asarray(d.get("cwm_OverSampling"), dtype=np.float64)
    osamp = np.where(np.isfinite(osamp) & (osamp > 0), osamp, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return w / (nev * osamp * d["_n_files"])


def _open_for_simweights(path):
    """simweights'e verilebilecek bir dosya nesnesi ac (pytables, yoksa h5py)."""
    try:
        return tables.open_file(path, "r"), "tables"
    except Exception:
        pass
    try:
        import h5py
        return h5py.File(path, "r"), "h5py"
    except Exception as e:
        raise RuntimeError("acilamadi: %s" % e)


def corsika_weight(d):
    """
    CORSIKA agirligi -- simweights + GaisserH3a (eski notebook ile ayni yontem).

    simweights HDF5'i DOGRUDAN okuyor (CorsikaWeightMap, PolyplopiaPrimary,
    I3CorsikaInfo ...), o yuzden dosyalari yeniden aciyoruz.  Sonuc
    Run/Event/SubEvent uzerinden geri eslestiriliyor -- satir sirasina
    guvenmiyoruz.

    NORMALIZASYON -- eski notebook'tan FARKLI ve bilerek:
      Eski kod her HDF5 icin nfiles=1 verip sonunda HDF5 sayisina
      boluyordu.  Bizim her parcamizda birden fazla L3 dosyasi var
      (--chunk-files), o yuzden parcanin KENDI n_l3_files'i nfiles olarak
      veriliyor ve sonda AYRICA bolme YAPILMIYOR.  Aksi halde agirliklar
      parca basina dosya sayisi kadar kucuk cikardi.
    """
    try:
        import simweights
    except ImportError:
        return _corsika_weight_manual(d)

    files = d.get("_files") or []
    n_l3 = d.get("_n_l3_per_file") or []
    if not files:
        print("  [!] dosya listesi yok -> yaklasik agirliga dusuluyor")
        return _corsika_weight_manual(d)

    key2w, n_fail = {}, 0
    for path, nl3 in zip(files, n_l3):
        if not nl3:
            print("  [!] %s: meta.json yok, nfiles bilinmiyor -> atlandi"
                  % os.path.basename(path))
            n_fail += 1
            continue
        fh = kind = None
        try:
            fh, kind = _open_for_simweights(path)
            wobj = simweights.CorsikaWeighter(fh, nfiles=nl3)
            w = np.asarray(wobj.get_weights(simweights.GaisserH3a()), dtype=np.float64)
            with tables.open_file(path, "r") as h5:
                eh = _table_nodes(h5)["I3EventHeader"]
                r, e, sub = _ids(eh)
            if len(w) != len(r):
                print("  [!] %s: simweights %d agirlik, %d olay -> atlandi"
                      % (os.path.basename(path), len(w), len(r)))
                n_fail += 1
                continue
            for k, ww in zip(zip(r, e, sub), w):
                key2w[k] = ww
        except Exception as ex:
            print("  [!] simweights basarisiz (%s): %s"
                  % (os.path.basename(path), str(ex)[:120]))
            n_fail += 1
        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    if not key2w:
        print("  [!] simweights hicbir dosyada calismadi -> yaklasik agirlik")
        return _corsika_weight_manual(d)

    out = np.array([key2w.get(k, np.nan) for k in
                    zip(d["Run"], d["Event"], d["SubEvent"])], dtype=np.float64)
    matched = np.isfinite(out).mean()
    print("  simweights + GaisserH3a: %d olay agirliklandirildi (%.0f%% eslesme%s)"
          % (len(key2w), 100 * matched,
             ", %d dosya basarisiz" % n_fail if n_fail else ""))
    if matched < 0.99:
        print("      [!] eslesmeyen olaylar NaN -> w_phys 0 sayilacak")
    return out


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
