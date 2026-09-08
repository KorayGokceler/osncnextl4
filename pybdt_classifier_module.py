'''
L4 siniflandirici uygulama modulu -- pybdt (AdaBoost) surumu.

l4_classifier_module.py'nin (LightGBM) BILINCLI ALTERNATIFI. bkz.
CLAUDE.md "BDT egitimi: pybdt kullanilacak" -- resmi oscNext yontemi
DEGIL, kullanicinin kendi tercihi.

Frame'den degisken okuma mantigi (FEATURE_MAP, read_feature) LightGBM
surumuyle ORTAK -- oradan import edilir. Boylece iki motor arasinda
hangi frame anahtarinin hangi degiskene karsilik geldigi TEK yerden
(l4_classifier_module.py) yonetilir, iki ayri kopya senkron kaymaz.

Model dosyasi: train_L4_classifier.py'nin .txt/.json'ununun aksine,
pybdt modeli SADECE pybdt kurulu bir ortamda okunabilen bir pickle'dir
(pybdt_train_L4_classifier.py --> util.save). Bu modul de dolayisiyla
SADECE pybdt'nin derlendigi ozel build'in env-shell'i icinde calisir.

KRITIK: pybdt icecube namespace'ine dahil DEGIL -- "import pybdt" /
"from pybdt import ml, util" kullanilir, "from icecube.pybdt import ..."
DEGIL (bkz. CLAUDE.md, bu yanlislik onceden derleme sorunu sanilmisti).

Kullanim:

    from pybdt_classifier_module import PyBDTL4Classifier, add_pybdt_L4_classifiers

    tray.Add(PyBDTL4Classifier, "noise_clf",
             ModelFile="models_pybdt/L4_noise_pybdt_model.pkl",
             OutputKey="L4_NoiseClassifier_ProbNu_pybdt")

    # ya da ikisi birden + kesim:
    add_pybdt_L4_classifiers(tray, "L4_cut_pybdt", model_dir="models_pybdt")
'''

import os
import json

import numpy as np
from icecube import icetray, dataclasses

from l4_classifier_module import FEATURE_MAP, read_feature

try:
    from pybdt import util
except ImportError:
    raise ImportError(
        "pybdt import edilemedi. Bu modul SADECE pybdt'nin derlendigi ozel "
        "build'in env-shell'i icinde calisir (bkz. CLAUDE.md). Dogru import "
        "'import pybdt' / 'from pybdt import ml, util' -- 'from icecube "
        "import pybdt' DEGIL.")


# ---------------------------------------------------------------------------
# Model yukleme
# ---------------------------------------------------------------------------

def load_model(model_file):
    '''
    pybdt modelini (.pkl) + yan JSON dosyasini yukle.

    Donen: (bdt_model, features, use_purity, sidecar)
    '''
    if not os.path.exists(model_file):
        raise IOError("Model dosyasi yok: %s" % model_file)

    bdt = util.load(model_file)

    json_file = os.path.splitext(model_file)[0] + ".json"
    if os.path.exists(json_file):
        with open(json_file) as fh:
            sidecar = json.load(fh)
        features = list(sidecar["features"])
        use_purity = bool(sidecar.get("use_purity", True))
    else:
        icetray.logging.log_warn(
            "pybdt_classifier: %s bulunamadi, degisken sirasi model "
            "nesnesinden aliniyor, use_purity=True varsayiliyor" % json_file)
        sidecar = {}
        features = list(bdt.feature_names)
        use_purity = True

    bn = list(bdt.feature_names)
    if bn and bn != features:
        raise ValueError(
            "Degisken sirasi uyusmuyor!\n  model: %s\n  json : %s"
            % (bn, features))

    unknown = [f for f in features if f not in FEATURE_MAP]
    if unknown:
        raise KeyError(
            "FEATURE_MAP'te tanimsiz degisken(ler): %s\n"
            "l4_classifier_module.py icindeki FEATURE_MAP'e ekleyin "
            "(iki motor da ayni haritayi kullanir)." % unknown)

    return bdt, features, use_purity, sidecar


# ---------------------------------------------------------------------------
# Tray modulu
# ---------------------------------------------------------------------------

class PyBDTL4Classifier(icetray.I3ConditionalModule):
    '''Egitilmis bir pybdt (AdaBoost) modelini frame bazinda uygular.'''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("ModelFile", "L4_*_pybdt_model.pkl yolu", None)
        self.AddParameter("OutputKey", "Yazilacak I3Double anahtari", None)
        self.AddParameter("MissingValue",
                          "Eksik degisken icin kullanilacak deger. "
                          "DIKKAT: pybdt LightGBM gibi NaN'i native "
                          "islemez -- karsilastirmalar NaN ile hep False "
                          "doner, bu da olayi ongorulemez sekilde bir "
                          "dala yonlendirir. Mumkunse 0.0 gibi makul bir "
                          "varsayilan kullanin.", 0.0)
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

        self.bdt, self.features, self.use_purity, self.sidecar = load_model(model_file)
        self.n_missing = {f: 0 for f in self.features}
        self.n_frames = 0

        print("PyBDTL4Classifier [%s]" % self.output_key)
        print("  model    : %s (pybdt/AdaBoost)" % model_file)
        print("  agac     : %d,  degisken: %d,  use_purity: %s" %
              (len(self.bdt), len(self.features), self.use_purity))
        if self.sidecar:
            print("  egitim   : %s" % self.sidecar.get("trained", "?"))
            print("  varsayilan kesim: %s (pybdt skor olcegi LightGBM ile "
                  "AYNI DEGIL)" % self.sidecar.get("default_cut", "?"))

    def Physics(self, frame):
        if self.output_key in frame:
            self.PushFrame(frame)
            return

        event = {}
        n_nan = 0
        for f in self.features:
            v = read_feature(frame, f)
            if not np.isfinite(v):
                self.n_missing[f] += 1
                n_nan += 1
                v = self.missing
            event[f] = v
        self.n_frames += 1

        if self.skip_incomplete and n_nan == len(self.features):
            self.PushFrame(frame)
            return

        score = float(self.bdt.score_event(event, use_purity=self.use_purity))
        frame[self.output_key] = dataclasses.I3Double(score)
        self.PushFrame(frame)

    def Finish(self):
        bad = {f: n for f, n in self.n_missing.items() if n > 0}
        if bad and self.n_frames:
            print("PyBDTL4Classifier [%s] eksik degisken raporu (%d frame):"
                  % (self.output_key, self.n_frames))
            for f, n in sorted(bad.items(), key=lambda kv: -kv[1]):
                frac = 100.0 * n / self.n_frames
                flag = "  <-- HEP EKSIK, model bozuk cikar" if frac > 99.9 else ""
                print("    %-30s %7d (%5.1f%%)%s" % (f, n, frac, flag))


# ---------------------------------------------------------------------------
# Kolaylik segmenti
# ---------------------------------------------------------------------------

NOISE_KEY = "L4_NoiseClassifier_ProbNu_pybdt"
MUON_KEY  = "L4_MuonClassifier_Data_ProbNu_pybdt"
CUT_KEY   = "L4_Cut_Bool_pybdt"


@icetray.traysegment
def add_pybdt_L4_classifiers(tray, name, model_dir,
                             noise_cut=0.5, muon_cut=0.5, apply_cut=True):
    '''
    Her iki pybdt siniflandiricisini ekle ve L4 kesimini hesapla.

    DIKKAT: noise_cut/muon_cut varsayilanlari (0.5) pybdt'nin skor
    olcegine gore SECILMEDI -- resmi bir referans yok (LightGBM
    surumundeki v00.07 degerleri 0.70/0.65 buraya TASINAMAZ).
    pybdt_train_L4_classifier.py'nin rate-vs-cut grafiginden kendi
    kesim degerinizi belirleyin.
    '''
    tray.Add(PyBDTL4Classifier, name + "_noise",
             ModelFile=os.path.join(model_dir, "L4_noise_pybdt_model.pkl"),
             OutputKey=NOISE_KEY)

    tray.Add(PyBDTL4Classifier, name + "_muon",
             ModelFile=os.path.join(model_dir, "L4_muon_pybdt_model.pkl"),
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
