# Kodun akış şeması — hangi dosya ne zaman çalışır

İki bağımsız aşama var. Birincisi IceTray gerektirir ve pahalıdır (bir kez);
ikincisi sadece numpy/pytables/pybdt ister ve sık tekrarlanır.

```
   AŞAMA A — İŞLEME (IceTray, saatler)      AŞAMA B — ANALİZ (dakikalar)
   ─────────────────────────────────────    ────────────────────────────
   L3 .i3.zst  ──►  L4 .hdf5                L4 .hdf5  ──►  .bdt model
```

---

# AŞAMA A — L3 `.i3` → L4 `.hdf5`

## Giriş noktası

```
notebook hücre "run_all(chunk_files=10)"
        │
        ▼
l4_run.run_all()                    her örnek için ayrı çubuk
        │
        ▼
l4_run.run_process(name)            komutu kurar
        │  subprocess.Popen([python, "-u", "process_L4.py", ...])
        │  stdout'u SATIR SATIR okur, [CHUNK]/[PROGRESS] ayrıştırır
        ▼
process_L4.py main()                ← BURADAN İTİBAREN AYRI SÜREÇ
```

`run_all` olmadan da çalışır — terminalden doğrudan `python process_L4.py …`.
Notebook sadece bir sürücü.

## `process_L4.py` — ana akış

```
1. icetray_env.require_icetray()       icecube import edilemezse SEBEBİNİ yazar
   icetray_env.get_I3Tray()            I3Tray'in yerini sürüme göre bulur
   icetray_env.load_deserialization_libs()
                                       simclasses/recclasses/genie_icetray…
                                       import edilmezse frame açılamaz

2. Girdi listesi                       --input glob'u ya da --input-list

3. validate_files()                    ÖN TARAMA (--scan quick)
        │                              her dosyanın ilk 25 frame'i okunur
        │                              bozuklar elenir → .badfiles.txt
        ▼
4. build_key_list()                    hangi frame anahtarları book edilecek
        │                              (örnek türüne göre: --genie/--noise/…)
        ▼
5. _run_tray(build_tray, …)            tray'i kurar ve çalıştırır
        │                              patlarsa bozuk dosyayı atıp YENİDEN dener
        ▼
6. _write_meta()                       <çıktı>.hdf5.meta.json
                                       n_l3_files ← ağırlık böleni
```

`--chunk-files N` verilirse 5–6 her parça için tekrarlanır; tamamlanmış
parçalar atlanır (çökme sonrası kaldığı yerden devam).

## Tray içinde modül sırası

Her frame bu zincirden **sırayla** geçer:

```
I3Reader                    GCD + .i3.zst dosyalarını okur
   │
_count_physics              [sayaç] Physics frame sayısı
   │
ProgressReporter            [sayaç] her 5000 frame'de bir [PROGRESS] satırı
   │
stream_filter               sub_event_stream == "InIceSplit" değilse ELE
   │
_count_stream               [sayaç] filtreden geçen
   │
oscNext_L4  ◄────────────── oscNext_L4_variables.py, asıl iş burada
   │
count                       [sayaç] book edilecek olay
   │
I3Writer                    (opsiyonel, --output-i3)
   │
add_booker  ◄────────────── simple_booker.py
                            hdfwriter varsa onu, yoksa SimpleBooker'ı kullanır
```

## `oscNext_L4` segmenti — L4 değişkenlerinin üretimi

```
oscNext_L4  (oscNext_L4_variables.py:809)
│
├─ PropagateGenieInfo              --genie ise: I3GenieInfo.n_flux_events
│                                  S frame'den P frame'e taşınır (ağırlık için)
│
├─ l3_cut                          L3_oscNext_bool ya da
│                                  IC2018_LE_L3_Full AND Data_quality_bool
│                                  GEÇMEYEN OLAY BURADA ELENİR
│
├─ oscNext_L4_common_variables
│     ├─ _first_hlc                ilk HLC hit → L4_first_hlc (I3Particle)
│     ├─ _add_rho_36               string 36'ya uzaklık → L4_first_hlc_rho ★
│     └─ _full_time_length_ratio   temizli/temizsiz süre oranı ★
│
├─ oscNext_L4_hit_statistics       (L3'te yoksa)
│     └─ common_variables          → cog_z ★, z_sigma ★, z_travel ★, n_hit_doms
│
├─ oscNext_L4_noise_cut_variables
│     ├─ I3StaticTWC               [-3.5, +4] µs statik pencere    ┐
│     ├─ I3SeededRTCleaning        SeededRT temizleme              │ micro_count
│     ├─ I3OMSelection             DeepCore fiducial DOM'lar       │ zinciri
│     ├─ I3TimeWindowCleaning      200 ns dinamik pencere          │
│     ├─ _micro_count              maks DOM sayısı → L4_micro_count ★
│     └─ I3FillRatioModule         → L4_fill_ratio ★
│
├─ oscNext_L4_atm_muon_classifier_variables
│     ├─ I3TensorOfInertia         → L4_ToI (aday, BDT girdisi değil)
│     ├─ linefit.simple            → L4_iLineFitParams.lf_vel ★
│     ├─ SmallQ_Box                (slc-veto yoksa atlanır)
│     ├─ _accumulated_time         yükün %75'ine varış süresi ★
│     ├─ _separation_in_cogs       (BDT girdisi değil)
│     └─ _vich                     veto bölgesi nedensel hitler → L4_VICH_nch ★
│
└─ compute_L4_cut                  SADECE --apply-cut ile
      └─ l4_classifier_module      eğitilmiş modeli frame'e uygular
```

★ = BDT girdisi (14 tane; gerisi L3'ten hazır geliyor ya da aday/tanı)

## Aşama A'da çalışan dosyalar

| Dosya | Rolü |
|---|---|
| `l4_run.py` | notebook sürücüsü (opsiyonel) |
| `process_L4.py` | ana script — tray'i kurar, çalıştırır, meta yazar |
| `oscNext_L4_variables.py` | **tüm L4 değişken hesabı** |
| `icetray_env.py` | icecube import köprüsü, DOM listeleri, pybdt kontrolü |
| `simple_booker.py` | `hdfwriter` yoksa fallback booker |
| `scan_files.py` | (opsiyonel, ayrı) bozuk dosya tarayıcı |

---

# AŞAMA B — L4 `.hdf5` → eğitilmiş model

IceTray **gerekmez**. Sadece `numpy`, `tables`, `pybdt`.

```
notebook bölüm 2-3
        │
        ├─ l4_data.dump_tables()        HDF5'te gerçekte hangi tablo/kolon var
        ├─ l4_data.check_registry()     REGISTRY'deki kolonlar bulundu mu
        └─ l4_data.check_feature_map()  REGISTRY ↔ FEATURE_MAP çakışması
                                        (AST ile okur, icetray gerekmez)
        │
notebook bölüm 4
        ▼
l4_data.load_sample(name, SAMPLES, WANTED)
        │
        ├─ load_one_file()              her HDF5 parçası için
        │     ├─ resolve_one()          ALTS ile kolon adını çöz
        │     │                         (lf_vel / LFVel / speed)
        │     ├─ Run/Event/SubEvent      ile tablolar EŞLEŞTİRİLİR
        │     └─ çözülemeyen → NaN + uyarı
        │
        └─ n_l3_files()                 .meta.json'dan ağırlık böleni
        ▼
   data = {"nue": {...}, "numu": {...}, "corsika": {...}, "noise": {...}}
        │
notebook bölüm 5
        ▼
l4_data.add_weights(data)               w_phys [Hz]
        ├─ genie_weight                 OneWeight·flux/n_flux/n_files
        ├─ noise_weight                 noise_weight·1e9/n_files
        ├─ corsika_weight               (simweights yok → yaklaşık)
        └─ Tablo 13 ile mertebe karşılaştırması
        │
notebook bölüm 6
        ▼
make_datasets()                         notebook içinde
        ├─ eğitim ağırlığı: sınıfları eşitle → [0,1]
        ├─ train/test %50 rastgele ayrım
        └─ pybdt.util.save() → 4 adet .ds dosyası
        │
notebook bölüm 7
        ▼
subprocess: pybdt_train.py              ← AYRI SÜREÇ
        ├─ ml.BDTLearner(features, w, w)
        ├─ learner.dtlearner            depth, min_split, num_cuts
        ├─ SameLeafPruner + CostComplexityPruner
        ├─ learner.train()              AdaBoost
        └─ çıktı: .bdt  .validator  .json  3 adet .png
        │
notebook bölüm 8-9
        ▼
Validator                               KS overtraining testi
scan_cut()                              rate-vs-cut, kesim seçimi
        │
notebook bölüm 10
        ▼
process_L4.py --apply-cut               ← AŞAMA A'YA GERİ DÖNER
        └─ pybdt_classifier_module.PyBDTClassifier
              └─ FEATURE_MAP (l4_classifier_module.py) ile frame'den okur
```

## Aşama B'de çalışan dosyalar

| Dosya | Rolü |
|---|---|
| `oscNext_L4_pybdt.ipynb` | arayüz — ince, mantık modüllerde |
| `l4_data.py` | REGISTRY, ALTS, yükleme, ağırlıklar, tutarlılık |
| `pybdt_train.py` | eğitim + doğrulama (ayrı süreç) |
| `pybdt_classifier_module.py` | modeli frame'e uygular |
| `l4_classifier_module.py` | `FEATURE_MAP` — frame'den okuma haritası |

---

# Çalışmayan dosyalar (referans)

| Dosya | Neden duruyor |
|---|---|
| `train_L4_classifier.py` | LightGBM yolu — teknik notun **resmi** yöntemi. pybdt'ye geçildi ama karşılaştırma gerekirse lazım. |
| `oscNext_L4_feature_engineering.ipynb` | eski LightGBM arayüzü. Yeni notebook'a **taşınmamış** bölümleri var: referansla karşılaştırma, türetilmiş değişkenler, data/MC uyumu, incremental feature scan. |
| `l4_classifier_module.py` | LightGBM uygulama modülü — ama `FEATURE_MAP`'i pybdt yolu da kullanıyor, yani **kısmen aktif**. |
| `simple_booker.py` | kendi build'de `hdfwriter` var, fallback devrede değil. cvmfs'e dönülürse gerekli. |
| `diagnose_env.py` | sorun çıkınca elle çalıştırılır. |

---

# İki kritik senkronizasyon

**1. `REGISTRY` (l4_data.py) ↔ `FEATURE_MAP` (l4_classifier_module.py)**

Birincisi HDF5 kolonunu, ikincisi frame anahtarını tarif ediyor. Ayrışırsa
eğitimde bir şey, uygulamada başka bir şey okunur — model **hata vermeden**
yanlış sonuç üretir. `check_feature_map()` bunu kontrol ediyor.

**2. `.meta.json` ↔ ağırlıklar**

`process_L4.py` yazar, `l4_data.load_sample` okur. Yoksa bölen yanlış olur
ve tüm oranlar kayar. Sidecar'ı olmayan eski HDF5'lerde uyarı basılır.
