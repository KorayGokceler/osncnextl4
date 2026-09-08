'''
L4 siniflandirici uygulama modulu -- pybdt.

pybdt_train.py'nin urettigi modeli (.bdt + .json) IceTray frame'lerine
uygular.  pybdt'nin kendi ornek modulunun (pybdt/python/pybdtmodule.py)
yaptigi isi yapar, ustune oscNext'e ozel iki sey ekler:
  * degiskenler frame'den FEATURE_MAP uzerinden okunur (varsfunc yerine)
  * eksik degisken sayaci + Finish() raporu

BAGIMLILIKLAR: pybdt + numpy + icecube.icetray/dataclasses.
sklearn / lightgbm KULLANILMAZ.

DIKKAT -- import yolu: "from pybdt import util", "from icecube import
pybdt" DEGIL (pybdt icecube namespace'ine dahil degildir, bkz. CLAUDE.md).

FEATURE_MAP (model degisken adi -> frame anahtari + kolon) su an
l4_classifier_module.py'den import edilir.  Bu, mapping'in TEK yerde
kalmasi icin bilincli bir tercih -- LightGBM'e ozgu bir sey degil, saf
frame okuma bilgisi (l4_classifier_module.py lightgbm'i sadece kendi
load_model() fonksiyonunun icinde import eder, dolayisiyla bu import
lightgbm gerektirmez).  Pipeline yeniden yazilirken mapping ayri bir
module tasinirsa burasi da guncellenmelidir.

Kullanim:

    from pybdt_classifier_module import PyBDTClassifier

    tray.Add(PyBDTClassifier, "noise_clf",
             ModelFile="models_pybdt/L4_noise.bdt",
             OutputKey="L4_NoiseClassifier_pybdt")
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
        "pybdt import edilemedi.  Bu modul SADECE pybdt'nin derlendigi "
        "build'in env-shell'i icinde calisir (bkz. CLAUDE.md).  Dogru "
        "import 'from pybdt import util' -- 'from icecube import pybdt' "
        "DEGIL.")


def load_model(model_file):
    '''
    pybdt modelini (.bdt) + yanindaki meta dosyasini (.json) yukle.

    Donen: (bdt, features, use_purity, meta)
    '''
    if not os.path.exists(model_file):
        raise IOError("Model dosyasi yok: %s" % model_file)

    bdt = util.load(model_file)

    json_file = os.path.splitext(model_file)[0] + ".json"
    if os.path.exists(json_file):
        with open(json_file) as fh:
            meta = json.load(fh)
        features = list(meta["features"])
        use_purity = bool(meta.get("use_purity", False))
    else:
        icetray.logging.log_warn(
            "pybdt_classifier: %s bulunamadi -- degisken sirasi modelden "
            "aliniyor, use_purity=False varsayiliyor.  Model use_purity=True "
            "ile egitildiyse skorlar YANLIS olur." % json_file)
        meta = {}
        features = list(bdt.feature_names)
        use_purity = False

    bn = list(bdt.feature_names)
    if bn and bn != features:
        raise ValueError(
            "Degisken sirasi uyusmuyor!\n  model: %s\n  json : %s"
            % (bn, features))

    unknown = [f for f in features if f not in FEATURE_MAP]
    if unknown:
        raise KeyError(
            "FEATURE_MAP'te tanimsiz degisken(ler): %s\n"
            "l4_classifier_module.py icindeki FEATURE_MAP'e ekleyin." % unknown)

    return bdt, features, use_purity, meta


class PyBDTClassifier(icetray.I3ConditionalModule):
    '''Egitilmis bir pybdt modelini frame bazinda uygular.'''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("ModelFile", "pybdt_train.py'nin urettigi .bdt", None)
        self.AddParameter("OutputKey", "Yazilacak I3Double anahtari", None)
        self.AddParameter("MissingValue",
                          "Eksik degisken icin kullanilacak deger.  DIKKAT: "
                          "pybdt NaN'i LightGBM gibi native islemez -- NaN ile "
                          "yapilan karsilastirmalar hep False doner ve olay "
                          "ongorulemez bir dala gider.  Makul bir sayisal "
                          "varsayilan verin.", 0.0)
        self.AddOutBox("OutBox")

    def Configure(self):
        model_file = self.GetParameter("ModelFile")
        self.output_key = self.GetParameter("OutputKey")
        self.missing = self.GetParameter("MissingValue")

        if not model_file or not self.output_key:
            raise ValueError("ModelFile ve OutputKey zorunlu")

        self.bdt, self.features, self.use_purity, self.meta = load_model(model_file)
        self.n_missing = {f: 0 for f in self.features}
        self.n_frames = 0

        print("PyBDTClassifier [%s]" % self.output_key)
        print("  model      : %s" % model_file)
        print("  agac       : %d,  degisken: %d" % (len(self.bdt), len(self.features)))
        print("  use_purity : %s" % self.use_purity)
        if self.meta:
            print("  egitim     : %s" % self.meta.get("trained", "?"))

    def Physics(self, frame):
        if self.output_key in frame:
            self.PushFrame(frame)
            return

        event = {}
        for f in self.features:
            v = read_feature(frame, f)
            if not np.isfinite(v):
                self.n_missing[f] += 1
                v = self.missing
            event[f] = v
        self.n_frames += 1

        score = float(self.bdt.score_event(event, use_purity=self.use_purity))
        frame[self.output_key] = dataclasses.I3Double(score)
        self.PushFrame(frame)

    def Finish(self):
        bad = {f: n for f, n in self.n_missing.items() if n > 0}
        if bad and self.n_frames:
            print("PyBDTClassifier [%s] eksik degisken raporu (%d frame):"
                  % (self.output_key, self.n_frames))
            for f, n in sorted(bad.items(), key=lambda kv: -kv[1]):
                frac = 100.0 * n / self.n_frames
                flag = "  <-- HEP EKSIK, model bozuk cikar" if frac > 99.9 else ""
                print("    %-30s %7d (%5.1f%%)%s" % (f, n, frac, flag))
