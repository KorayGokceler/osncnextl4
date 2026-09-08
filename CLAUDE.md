# oscNext L4 — proje bilgilendirmesi

Bu dosya her Claude Code session'ında otomatik yüklenir. Amacı: kod tabanını
hiç görmemiş bir session'ın, fizik/IceTray bağlamını ve mevcut riskleri hızlıca
kavrayıp anlamlı fikir yürütebilmesi.

## Ne yapıyor bu repo

IceCube deneyinin oscNext (düşük enerji nötrino) analizinde **Level 3 → Level
4** işleme adımını yeniden inşa ediyor. L3 çıktısı `.i3` dosyalarından L4
ayırt edici değişkenlerini hesaplıyor, HDF5'e "book" ediyor, feature
engineering yapıyor ve iki LightGBM BDT'si eğitiyor:

- **Noise BDT**: saf gürültü (vuvuzela) reddi
- **Muon BDT**: atmosferik muon reddi (arka plan = CORSIKA)

Referans: oscNext technical note v00.07 (bölüm 3.4–3.6, Tablo 10–12).

## Neden yeniden yazıldı (kritik bağlam)

Kullanılan IceTray meta-projesinde (`py3-v4.4.2`) şu satırlar **yok**:

- `icecube.oscNext` projesi tamamen yok → `I3Classifier` (model uygulama),
  `oscNext_cut`, `calc_rho_36` kullanılamıyor.
- `icecube.hdfwriter` yok (meta-proje HDF5 destesiz derlenmiş) →
  `simple_booker.py` pytables ile fallback booking yapıyor.
- Eski proje bağımlılıkları yok: `tau_bdt.I3CutL7Module` (VICH),
  `analysis.event_selection` (Dunkman değişkenleri: accumulated_time,
  separation_in_cogs), `slc-veto` (QR box, opsiyonel).

Bu yüzden `oscNext_L4_variables.py` içinde VICH, accumulated_time,
separation_in_cogs saf Python'la **teknik nottaki tanımlara göre tahminen**
yeniden yazıldı ve **henüz orijinal C++ implementasyonuyla doğrulanmadı**.
Herhangi bir session bu konuda ilerleme kaydedebilir/fikir üretebilirse
değerli olur — bkz. "Açık riskler" altında.

## Dosya haritası ve veri akışı

```
.i3 (L3 çıktısı, pass3)
   │
   ▼
process_L4.py  ──uses──►  oscNext_L4_variables.py (oscNext_L4 traysegment)
   │                          │
   │                          ├─ common: first_hlc, rho_36, FullTimeLengthRatio
   │                          ├─ muon vars: ToI, iLineFit, VICH, accumulated_time,
   │                          │             separation_in_cogs
   │                          ├─ noise vars: micro_count, fill_ratio
   │                          └─ hit_statistics: cog_z, z_sigma, z_travel, n_hit_doms
   │
   ├─uses──► simple_booker.py (add_booker: hdfwriter varsa o, yoksa SimpleBooker)
   │
   ▼
.hdf5  (L4_output/hdf5/<sample>/L4_*.hdf5)
   │
   ▼
oscNext_L4_feature_engineering.ipynb  (ANA ARAYÜZ — 11 bölüm, sırayla çalıştırılır)
   0  Konfigürasyon
   1  process_L4.py'yi notebook'tan çalıştırma (+smoke test)
   2  Booking doğrulaması
   3  Feature registry (tek doğruluk kaynağı — hangi kolon, hangi tablo)
   4  Yükleme (HDF5 → DataFrame)
   5  Sağlık kontrolü (eksik/NaN kolonlar)
   6  Yeniden yazılan değişkenlerin referansla karşılaştırılması
   7  Livetime ve ağırlıklar
   8  Türetilmiş değişkenler
   9  Data/MC uyum kontrolü
  10  Korelasyon + feature importance + incremental scan (~40 değişken taraması)
  11  Export → parquet (L4_noise_training.parquet, L4_muon_training.parquet)
   │
   ▼
train_L4_classifier.py  (native LightGBM API, sklearn API DEĞİL)
   │
   ▼
L4_{tag}_model.txt + .json  (model + sidecar: değişken sırası + sınıf haritası)
   │
   ▼
l4_classifier_module.py  (L4Classifier tray modülü — I3Classifier'in yerine)
   FEATURE_MAP: model değişken adı → (frame anahtarı, kolon adı)
```

Yardımcı/tanı scriptleri: `diagnose_env.py` (ortamda ne var/yok),
`dump_columns.py` (üretilen HDF5'in tam kolon isimleri),
`inspect_classifier_api.py` (referans model şemasını çözer, eğitim
scriptini yazmadan önce çalıştırılmalı).

## Kritik senkronizasyon noktası

`FEATURE_MAP` (`l4_classifier_module.py`) ile notebook'taki **feature
registry** (bölüm 3) birbiriyle satır satır aynı olmalı. Biri diğerinden
farklı bir kolon okursa model **sessizce** yanlış tahmin üretir — hata
fırlatmaz. Bu iki tanımı karşılaştırmadan model/kolon değişikliği
önerilmemeli.

Model formatı bilinçli olarak joblib/pickle değil, LightGBM native metin
formatı (`.txt`) + JSON sidecar: IceTray ortamında sklearn/joblib yok, sadece
`lightgbm` + `numpy` var.

## Mevcut durum (README'den)

- [x] Ortam doğrulandı (`hdfwriter` yok → SimpleBooker; `oscNext` projesi yok;
      `slc-veto` yok)
- [x] νe işleme çalışıyor, tüm tablolar book ediliyor
- [ ] νμ / CORSIKA / noise işleme henüz denenmedi
- [ ] Sütun isimleri kesinleştirilmedi
- [ ] Yeniden yazılan değişkenler (VICH, accumulated_time, separation_in_cogs)
      referansla doğrulanmadı
- [ ] Ağırlıklar doğrulanmadı
- [ ] Model henüz eğitilmedi

## Açık riskler / fikir yürütülebilecek noktalar

1. **VICH tanımı** (`_vich` fonksiyonu, `oscNext_L4_variables.py:458`) —
   veto bölgesi hit'lerinin COG vertex'ine göre nedensellik hızı [0.25, 0.4]
   m/ns aralığıyla işaretleniyor. Orijinal `tau_bdt.I3CutL7Module`
   mevcut olmadığı için birebir doğrulama yapılamadı. Teknik nottaki
   tanımla (bölüm 3.4) satır satır karşılaştırma faydalı olur.
2. **accumulated_time / separation_in_cogs** (Dunkman değişkenleri) —
   orijinal `analysis.event_selection.CalculateVariables` modülü yok;
   tanım teknik nottan (Tablo 12) çıkarıldı. `separation_in_cogs` BDT
   girdisi değil (kritik değil) ama `accumulated_time` muon BDT'sinde
   kullanılıyor — yanlışsa model performansını doğrudan etkiler.
3. **FullTimeLengthRatio yönü** — kod "temizlenmiş/temizlenmemiş" (~1 iyi
   olay, ~0 gürültü) varsayıyor; teknik notun Şekil 13'üyle yön tutarlılığı
   gerçek veri üzerinde tekrar kontrol edilmeli.
4. **Ağırlık zinciri** (`PropagateGenieInfo`, `process_L4.py` MC_KEYS/
   NOISE_MC_KEYS/CORSIKA_KEYS) — pass3'te I3GenieInfo yoksa NEvents*fraksiyon
   fallback'ine düşülüyor; bunun ne sıklıkla tetiklendiği ve ne kadar sapma
   yarattığı ölçülmedi.
5. **fill_ratio yarıçapı** (`FILL_RATIO_SPHERICAL_RADIUS_MEAN = 1.6`) — GRECO
   için optimize edilmiş, oscNext için yeniden ayarlanmadı. Bölüm 10'daki
   feature importance / incremental scan sonuçlarına göre yeniden
   optimize edilmesi gündeme gelebilir.
6. **ντ ve gerçek dedektör verisi yok** — sinyal tanımı νe+νμ (ντ CC ~%3),
   muon BDT arka planı CORSIKA (gerçek veri değil). Bu ikame ne kadar
   sapma yaratıyor, data/MC uyum kontrolü (bölüm 9) devreye girince
   netleşecek.

## Referans belgeler (`reference/`)

- `reference/OscNext_v00.074_pass2_technical_note.pdf` — pass2 için resmi
  oscNext teknik notu (83 sayfa). **pass3 için değil**, ama L4 mantığının
  büyük kısmı (değişken tanımları, BDT hiperparametreleri) pass3'te de
  aynı kabul ediliyor.
- `reference/pass3_L3_process.py` — kullanıcının elindeki **gerçek pass3 L3
  işleme scripti** (GRECO `grecovariables.DeepCoreCleaning`/`DeepCoreCuts`
  kullanıyor). `oscNext_L4_variables.py`'nin varsaydığı L3 çıktısıyla
  karşılaştırıldı ve **doğrulandı**:
  - `SRTTWSplitInIcePulsesDC` → `CLEANED_PULSES_DEFAULT` ile birebir aynı
  - `L3_oscNext_bool = IC2018_LE_L3_bools["IC2018_LE_L3_Full"] AND
    Data_quality_bool` → `oscNext_L4_variables.py`'deki `l3_cut`
    fonksiyonuyla birebir aynı mantık
  - SLOP filtresi / LID errata veri kalitesi kesimi de kod yorumundaki
    varsayımla eşleşiyor

## BDT eğitimi: pybdt kullanılacak (bilinçli sapma — dikkat)

`icecube/icetray` (private) içindeki `pybdt` (IceCube'un kendi AdaBoost
tabanlı BDT kütüphanesi, C++/boost_python; kod repoda `pybdt/` altında)
bu projede **kasıtlı olarak** kullanılacak.

**Bunun resmi oscNext analiziyle uyuşmadığını bil:** Teknik not (pass2,
v00.074, bölüm 3.6.1) L4 noise/muon sınıflandırıcılarının LightGBM
(gradient boosting) ile eğitildiğini açıkça yazıyor, pybdt hiç
geçmiyor; Tablo 10'daki hiperparametreler de (`max_depth`, `num_leaves`,
`max_bin`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`) LightGBM'in
native isimleri. `train_L4_classifier.py` şu an bu resmi yaklaşımı
(LightGBM) doğru şekilde uyguluyor ve referans olarak repoda duruyor.

pybdt'ye geçiş şu sonuçları doğurur:
- pybdt AdaBoost yapıyor, LightGBM'in gradient-boosting + leaf/lambda
  regularizasyon mantığı yok — Tablo 10 parametreleri pybdt'ye
  **doğrudan taşınamaz**, pybdt'nin kendi API'sine göre (`num_trees`,
  `beta`, `depth`, `min_split`, `prune_strength`, `use_purity`) ayrıca
  ayarlandı — bkz. `pybdt_train_L4_classifier.py` içindeki `PYBDT_PARAMS`
  (pybdt'nin kendi örnek/varsayılan değerleri, oscNext için optimize
  EDİLMEDİ, sadece başlangıç noktası).
- pybdt derlenmiş bir C++/boost_python eklentisi. **py3-v4.4.2 cvmfs
  dağıtımında pybdt YOK** (`BUILD_PYBDT` bayrağı kapalı gelmiş,
  `pybdt/CMakeLists.txt`'te `USE_TOOLS ... gsl` gerektiriyor).
  **Çözüldü:** `icecube/icetray` kaynağı (v1.17.0, ZIP indirilip) ayrı bir
  build dizininde (`/data/user/<kullanıcı>/icetray_build/`) `cmake
  -DBUILD_PYBDT=ON` ile yapılandırılıp `make pybdt` ile derlendi (GSL 2.8
  cvmfs'te zaten mevcuttu, cmake buldu). Build sadece
  `serialization`/`icetray`/`dataclasses`/`pybdt` hedeflerini derledi
  (tüm meta-proje değil), birkaç dakika sürdü. `/cvmfs` salt-okunur
  olduğu için build cvmfs'e hiç yazmadı, tamamen ayrı bir dizinde.
- **KRİTİK — import yolu diğer IceTray projelerinden FARKLI:** pybdt,
  `icecube` isim alanına dahil DEĞİL — `from icecube import pybdt`
  ÇALIŞMAZ (bu yüzden başta "pybdt derlenmemiş" sanılıp gereksiz yere
  şüpheye düşüldü). Doğrusu: `import pybdt` / `from pybdt import ml,
  util` (pybdt'nin kendi kaynak kodu da bunu kullanıyor, bkz.
  `pybdt/python/pybdtmodule.py`). `l4_classifier_module.py` ve
  `train_L4_classifier.py`'nin aksine pybdt `icecube.*` namespace
  paketi değil, bağımsız üst düzey bir pip-tarzı pakettir.
- Bu özel build'in ortamı: `eval $(/cvmfs/.../py3-v4.4.2/setup.sh)` →
  `cd <build_dizini> && ./env-shell.sh`. Jupyter de bu ortamdan
  başlatılmalı (Jupyter içindeki terminaller/kernel'ler ortamı miras
  alır, tekrar env-shell gerekmez; sadece Jupyter sunucusu yeniden
  başladığında bu iki adım tekrarlanır).
- Eğitim + uygulama çifti yazıldı:
  - `pybdt_train_L4_classifier.py` — `train_L4_classifier.py` ile aynı
    parquet girdisini kullanır (`load_data` oradan import edilir),
    `pybdt.ml.BDTLearner` ile eğitir, `.pkl` + `.json` sidecar kaydeder.
  - `pybdt_classifier_module.py` — `l4_classifier_module.py`'deki
    `FEATURE_MAP`/`read_feature`'ı yeniden kullanır (tek doğruluk
    kaynağı korunur), `PyBDTL4Classifier` tray modülüyle `.pkl` modeli
    frame'e uygular.
  - **Henüz yapılmadı:** gerçek eğitim verisi yok (notebook'un
    parquet export adımı — bölüm 11 — hiç çalıştırılmadı), bu yüzden
    `pybdt_train_L4_classifier.py` uçtan uca test edilmedi.

## Konvansiyonlar

- Kod ve yorumlar Türkçe.
- Veri repoya girmez (`.gitignore`: `L4_output/`, model/veri uzantıları).
- Notebook commit'lenmeden önce `nbstripout` ile temizlenmeli (çıktı hücreleri
  MB'larca yer kaplar ve anlamsız diff üretir).
- `oscNext_L4_variables.py` içindeki her yeniden yazılmış fonksiyonun
  docstring'inde orijinalin nereden geldiği ve neden değiştiği yazılı —
  değişiklik yapmadan önce bu docstring'leri okuyun.
