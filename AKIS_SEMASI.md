# Akış şeması — hangi dosya ne zaman çalışır

İki aşama var: **L3 → L4 işleme** (IceTray gerekir, saatler sürer) ve
**eğitim** (IceTray gerekmez, dakikalar sürer). Arayüz `oscNext_L4.ipynb`.

## Aşama 1 — L3 → L4 işleme

```
L3 .i3.zst dosyalari
   │
   ├─ scan_files.py            bozuk dosyalari ele, saglam liste uret
   │                           (set basina BIR KEZ; sonra --scan off)
   ▼
process_L4.py                  surucu: tray kurar, calistirir, book eder
   │   │
   │   ├─ icetray_env.py       TUM icecube import'lari buradan gecer
   │   │
   │   ├─ oscNext_L4_variables.py     tray segment'leri:
   │   │      oscNext_L4                 ana segment
   │   │      ├─ common      first_hlc, rho_36, FullTimeLengthRatio
   │   │      ├─ muon vars   ToI, iLineFit, VICH, accumulated_time
   │   │      ├─ noise vars  micro_count, fill_ratio
   │   │      └─ hit_stats   cog_z, z_sigma, z_travel, n_hit_doms
   │   │
   │   └─ simple_booker.py     add_booker: hdfwriter varsa o,
   │                           yoksa pytables fallback (cvmfs icin)
   ▼
L4_output/hdf5/<ornek>/L4_*.hdf5   + <cikti>.meta.json
```

`l4_run.py` bu adımın sürücüsü: `configure_runner`, `run_process`,
`run_all`, `run_process_parallel` (+ canlı ilerleme çubuğu). Notebook
bölüm 1 bunu çağırır.

## Aşama 2 — Eğitim

```
L4_*.hdf5
   │
   ├─ l4_data.py               REGISTRY/ALTS, dump_tables, check_registry,
   │                           check_feature_map, load_sample, add_weights
   ▼
numpy dizileri  (notebook bolum 4-5)
   │
   ▼
L4_output/ds/L4_<tag>_dataset.npz   (notebook bolum 6)
   │
   ▼
train_L4_classifier.py         LightGBM, Tablo 10 hiperparametreleri
   │
   ▼
L4_output/models/
   L4_<tag>_model.txt          LightGBM native metin formati
   L4_<tag>_model.json         degisken sirasi + metrikler + meta
   <tag>_cuts.png  <tag>_overtrain.png  <tag>_dist.png
   │
   ▼
l4_classifier_module.py        L4Classifier tray modulu
                               add_L4_classifiers(tray, ...) -> frame'e I3Double
```

## Tray modül sırası (`oscNext_L4` segmenti içinde)

1. L3 kesimi (`l3_cut`) — `IC2018_LE_L3_Full AND Data_quality_bool`
2. `first_hlc` → `L4_FirstHLC`
3. `rho_36`, `FullTimeLengthRatio`
4. muon değişkenleri (`iLineFit`, VICH, `accumulated_time`, opsiyonel ToI)
5. noise değişkenleri (`micro_count` zinciri, `fill_ratio`)
6. hit istatistikleri (L3 bunları sildiği için yeniden hesaplanır)
7. ağırlık zinciri (`PropagateGenieInfo`)

## Yardımcı / tanı

| dosya | ne zaman |
|---|---|
| `setup_env.sh` | ortamı bul, shell aç, tek komut çalıştır, Jupyter kernel kaydet |
| `icetray_env.py` | her `icecube` import'u; `get_I3Tray`, `require_lightgbm` |
| `diagnose_env.py` | "import edilmiyor" dediğinde: ortamda ne var/yok |
| `scan_files.py` | bozuk `.i3.zst` tara, `--good-list` üret |

## Referans (`reference/`) — çalışmaz, kaynaktır

| dosya | ne |
|---|---|
| `OscNext_v00.074_pass2_technical_note.pdf` | resmi teknik not (Tablo 10–13) |
| `oscNext_L4_pass2_original.py` | orijinal L4 tray segment'i (Tom Stuttard) |
| `pass3_L3_process.py` | kullanıcının gerçek pass3 L3 işleme scripti |
