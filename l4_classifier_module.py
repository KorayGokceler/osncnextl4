'''
L4 siniflandirici uygulama modulu -- icecube.oscNext.tools.classifier.I3Classifier
yerine gecer.

oscNext projesi meta-projede bulunmadigi icin I3Classifier kullanilamiyor.  Bu
modul ayni isi yapar:  modeli yukle -> frame'den degiskenleri oku -> tahmin et
-> I3Double olarak frame'e yaz.

BAGIMLILIKLAR: sadece lightgbm + numpy.
IceTray ortaminda (py3-v4.4.2) sklearn ve joblib YOK, o yuzden model native
metin formatinda okunur:

    L4_noise_model.txt    LightGBM agaclari
    L4_noise_model.json   degisken listesi (SIRA ONEMLI) + sinif haritasi

Kullanim:

    from l4_classifier_module import L4Classifier, add_L4_classifiers

    tray.Add(L4Classifier, "noise_clf",
             ModelFile="models/L4_noise_model.txt",
             OutputKey="L4_NoiseClassifier_ProbNu")

    # ya da ikisi birden + kesim:
    add_L4_classifiers(tray, "L4_cut", model_dir="models")
'''

import os
import json

import numpy as np
from icecube import icetray, dataclasses


# ---------------------------------------------------------------------------
# Frame'den degisken okuma
# ---------------------------------------------------------------------------
#
# Modelin degisken isimleri ("NchCleaned", "cog_z", "micro_count", ...) ile
# frame'deki konumlari ("IC2018_LE_L3_Vars"["NchCleaned"], ...) arasindaki
# harita.  Notebook'taki feature registry ile AYNI olmali.
#
# Bu haritanin notebook ile senkron kalmasi kritik: egitimde bir kolon,
# uygulamada baska bir kolon okunursa model sessizce sacmalar.

L3V = "IC2018_LE_L3_Vars"
# L3 (grecovariables.DeepCoreCleaning) "SRTTWSplitInIcePulsesDC" uretir.
CLEANED_PULSES = "SRTTWSplitInIcePulsesDC"
HITSTAT = CLEANED_PULSES + "HitStatistics"
HITMULT = CLEANED_PULSES + "HitMultiplicity"

FEATURE_MAP = {
    # --- noise BDT ---
    "NchCleaned":          (L3V, "NchCleaned"),
    "micro_count":         ("L4_micro_count", "STW_m3500p4000_DTW200"),
    "iLineFit_speed":      ("L4_iLineFitParams", "lf_vel"),
    "fill_ratio":          ("L4_fill_ratio", "fill_ratio_from_mean"),
    # pass3 L3 map'inde oran YOK -> L4 tray'inde hesaplanip yaziliyor
    "FullTimeLengthRatio": ("L4_FullTimeLengthRatio", "value"),

    # --- muon BDT ---
    "ICVetoHits":       (L3V, "ICVetoHits"),
    "RTVeto250Hits":    (L3V, "RTVeto250Hits"),
    "NAbove200Hits":    (L3V, "NAbove200Hits"),
    "VICH_nch":         ("L4_VICH_nch", "value"),
    "accumulated_time": ("L4_accumulated_time", "value"),
    "first_hlc_rho":    ("L4_first_hlc_rho", "value"),
    "cog_z":            (HITSTAT, "cog_z"),
    "z_sigma":          (HITSTAT, "z_sigma"),   # yoksa cog_z_sigma -- read_feature dener
    "z_travel":         (HITSTAT, "z_travel"),

    # --- aday / turetilmis girdiler icin ---
    "VICH_npulses":       ("L4_VICH_npulses", "value"),
    "VICH_qtot":          ("L4_VICH_qtot", "value"),
    "separation_in_cogs": ("L4_separation_in_cogs", "value"),
    "n_hit_doms":         (HITMULT, "n_hit_doms"),
    "cog_x":              (HITSTAT, "cog_x"),
    "cog_y":              (HITSTAT, "cog_y"),
    "z_min":              (HITSTAT, "z_min"),
    "z_max":              (HITSTAT, "z_max"),
    "CausalVetoHits":     (L3V, "CausalVetoHits"),
    "C2HR6":              (L3V, "C2HR6"),
    "VertexGuessZ":       (L3V, "VertexGuessZ"),
    "DCFiducialHits":     (L3V, "DCFiducialHits"),
    "STW9000_DTW300Hits": (L3V, "STW9000_DTW300Hits"),
    "CleanedFullTimeLength":   (L3V, "CleanedFullTimeLength"),
    "UncleanedFullTimeLength": (L3V, "UncleanedFullTimeLength"),
    "VetoFiducialRatioHits":   (L3V, "VetoFiducialRatioHits"),
    "first_hlc_x": ("L4_first_hlc", "x"),
    "first_hlc_y": ("L4_first_hlc", "y"),
    "first_hlc_z": ("L4_first_hlc", "z"),
}


# Sutun adi surumlere gore degisebilen alanlar icin alternatifler
COLUMN_ALTS = {
    ("L4_iLineFitParams", "lf_vel"): ["LFVel", "speed"],
    (HITSTAT, "z_sigma"):            ["cog_z_sigma"],
    ("L4_fill_ratio", "fill_ratio_from_mean"): ["fillratio_from_mean"],
}


def _get_col(obj, col):
    if hasattr(obj, "keys"):
        return float(obj[col]) if col in obj else None
    if col == "value":
        return float(obj.value)
    if hasattr(obj, col):
        return float(getattr(obj, col))
    return None


def read_feature(frame, name):
    '''Frame'den tek bir degiskeni oku.  Bulunamazsa NaN dondur.'''
    loc = FEATURE_MAP.get(name)
    if loc is None:
        return np.nan
    key, col = loc
    if key not in frame:
        return np.nan
    obj = frame[key]
    for candidate in [col] + COLUMN_ALTS.get((key, col), []):
        try:
            v = _get_col(obj, candidate)
        except (KeyError, AttributeError, TypeError, ValueError):
            v = None
        if v is not None:
            return v
    return np.nan


# ---------------------------------------------------------------------------
# Model yukleme
# ---------------------------------------------------------------------------

def load_model(model_file):
    '''
    Native LightGBM modeli + JSON yan dosyasini yukle.

    model_file .txt yolu; yanindaki .json otomatik bulunur.
    Donen: (booster, features, sidecar)
    '''
    import lightgbm as lgb

    if not os.path.exists(model_file):
        raise IOError("Model dosyasi yok: %s" % model_file)

    booster = lgb.Booster(model_file=model_file)

    json_file = os.path.splitext(model_file)[0] + ".json"
    if os.path.exists(json_file):
        with open(json_file) as fh:
            sidecar = json.load(fh)
        features = list(sidecar["features"])
    else:
        # Yan dosya yoksa modelin kendi degisken isimlerine guven
        icetray.logging.log_warn(
            "l4_classifier: %s bulunamadi, degisken sirasi booster'dan aliniyor"
            % json_file)
        sidecar = {}
        features = list(booster.feature_name())

    # Tutarlilik kontrolu -- sessiz hatanin en olasi kaynagi burasi
    bn = list(booster.feature_name())
    if bn and bn != features:
        raise ValueError(
            "Degisken sirasi uyusmuyor!\n  booster: %s\n  json   : %s"
            % (bn, features))

    unknown = [f for f in features if f not in FEATURE_MAP]
    if unknown:
        raise KeyError(
            "FEATURE_MAP'te tanimsiz degisken(ler): %s\n"
            "l4_classifier_module.py icindeki FEATURE_MAP'e ekleyin." % unknown)

    return booster, features, sidecar


# ---------------------------------------------------------------------------
# Tray modulu
# ---------------------------------------------------------------------------

class L4Classifier(icetray.I3ConditionalModule):
    '''Egitilmis bir LightGBM modelini frame bazinda uygular.'''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("ModelFile", "L4_*_model.txt yolu", None)
        self.AddParameter("OutputKey", "Yazilacak I3Double anahtari", None)
        self.AddParameter("MissingValue",
                          "Eksik degisken icin kullanilacak deger "
                          "(NaN -> LightGBM kendi isler)", np.nan)
        self.AddParameter("SkipIfIncomplete",
                          "Degiskenlerin hepsi eksikse frame'i atla", False)
        self.AddOutBox("OutBox")

    def Configure(self):
        model_file = self.GetParameter("ModelFile")
        self.output_key = self.GetParameter("OutputKey")
        self.missing = self.GetParameter("MissingValue")
        self.skip_incomplete = self.GetParameter("SkipIfIncomplete")

        if not model_file or not self.output_key:
            raise ValueError("ModelFile ve OutputKey zorunlu")

        self.booster, self.features, self.sidecar = load_model(model_file)
        self.n_missing = {f: 0 for f in self.features}
        self.n_frames = 0

        print("L4Classifier [%s]" % self.output_key)
        print("  model    : %s" % model_file)
        print("  agac     : %d,  degisken: %d" %
              (self.booster.num_trees(), len(self.features)))
        if self.sidecar:
            print("  egitim   : %s (lightgbm %s)" %
                  (self.sidecar.get("trained", "?"),
                   self.sidecar.get("lightgbm_version", "?")))
            print("  varsayilan kesim: %s" % self.sidecar.get("default_cut", "?"))

    def Physics(self, frame):
        if self.output_key in frame:
            self.PushFrame(frame)
            return

        x = np.empty((1, len(self.features)), dtype=np.float64)
        n_nan = 0
        for i, f in enumerate(self.features):
            v = read_feature(frame, f)
            if not np.isfinite(v):
                self.n_missing[f] += 1
                n_nan += 1
                v = self.missing
            x[0, i] = v
        self.n_frames += 1

        if self.skip_incomplete and n_nan == len(self.features):
            self.PushFrame(frame)
            return

        # binary objective -> predict dogrudan P(pozitif sinif)
        prob = float(self.booster.predict(x)[0])
        frame[self.output_key] = dataclasses.I3Double(prob)
        self.PushFrame(frame)

    def Finish(self):
        bad = {f: n for f, n in self.n_missing.items() if n > 0}
        if bad and self.n_frames:
            print("L4Classifier [%s] eksik degisken raporu (%d frame):"
                  % (self.output_key, self.n_frames))
            for f, n in sorted(bad.items(), key=lambda kv: -kv[1]):
                frac = 100.0 * n / self.n_frames
                flag = "  <-- HEP EKSIK, model bozuk cikar" if frac > 99.9 else ""
                print("    %-30s %7d (%5.1f%%)%s" % (f, n, frac, flag))


# ---------------------------------------------------------------------------
# Kolaylik segmenti
# ---------------------------------------------------------------------------

NOISE_KEY = "L4_NoiseClassifier_ProbNu"
MUON_KEY  = "L4_MuonClassifier_Data_ProbNu"
CUT_KEY   = "L4_Cut_Bool"


@icetray.traysegment
def add_L4_classifiers(tray, name, model_dir,
                       noise_cut=0.70, muon_cut=0.65, apply_cut=True):
    '''
    Her iki siniflandiriciyi ekle ve L4 kesimini hesapla.

    Kesim degerleri v00.07 referansi; kendi modelinizin optimal degerini
    train_L4_classifier.py'nin rate-vs-cut grafiginden secin.
    '''
    tray.Add(L4Classifier, name + "_noise",
             ModelFile=os.path.join(model_dir, "L4_noise_model.txt"),
             OutputKey=NOISE_KEY)

    tray.Add(L4Classifier, name + "_muon",
             ModelFile=os.path.join(model_dir, "L4_muon_model.txt"),
             OutputKey=MUON_KEY)

    if apply_cut:
        def overall_cut(frame):
            if NOISE_KEY not in frame or MUON_KEY not in frame:
                frame[CUT_KEY] = icetray.I3Bool(False)
                return True
            ok = (frame[NOISE_KEY].value >= noise_cut and
                  frame[MUON_KEY].value >= muon_cut)
            frame[CUT_KEY] = icetray.I3Bool(bool(ok))
            return True
        tray.Add(overall_cut, name + "_overall_cut")
