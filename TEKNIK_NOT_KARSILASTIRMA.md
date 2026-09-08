# HDF5'e ne yazıyoruz + teknik notla satır satır karşılaştırma

Kaynak: `reference/OscNext_v00.074_pass2_technical_note.pdf` (pass2, 83 sayfa).
İlgili bölümler: §3.4 (DeepCore Filter), §3.5.1 (L3 değişkenleri, Tablo 7–8),
§3.6 (L4, Tablo 10–13).

> **Not pass2 için.** Bizim girdimiz pass3. L4 mantığının büyük kısmı aynı
> kabul ediliyor ama isimlendirme değişmiş (aşağıda işaretli).

---

# Bölüm 1 — HDF5'e tam olarak ne yazıyoruz

`process_L4.py` bir **anahtar listesi** kuruyor (`build_key_list`) ve booker
her anahtarı **ayrı bir HDF5 tablosuna** çeviriyor. Tablo adı = frame anahtarı.
Her tabloda ortak indeks: `Run`, `Event`, `SubEvent`.

Frame nesnesi tipine göre kolonlaşma (`simple_booker.extract_scalars`,
hdfwriter da aynı şemayı üretiyor):

| Frame tipi | HDF5'te |
|---|---|
| `I3Double` / `I3Int` / `I3Bool` | tek kolon: `value` |
| `I3MapStringDouble` / `I3MapStringInt` | **her map anahtarı ayrı kolon** |
| `I3Particle` | `x, y, z, time, energy, length, speed, zenith, azimuth` |
| `I3LineFitParams` | `lf_vel`, `lf_vel_x/y/z`, `n_hits` (pass3 adları) |
| `I3HitStatisticsValues` | `cog_x/y/z`, `z_min`, `z_max`, `z_mean`, `z_sigma`, `z_travel`, … |
| `I3HitMultiplicityValues` | `n_hit_strings`, `n_hit_doms`, `n_hit_doms_one_pulse`, `n_pulses` |
| `I3FillRatioInfo` | `fill_ratio_from_mean`, `fill_radius_from_mean`, … |
| `I3EventHeader` | `run_id`, `event_id`, … + `time_start_mjd_day/sec/ns` |

**Pulse serileri book EDİLMİYOR.** 5000 pulse'lık bir olay düz tabloya
sığmaz. Sadece onlardan hesaplanan özet sayılar yazılıyor — BDT'nin
ihtiyacı olan da bu.

## Yazılan anahtarlar

### Her zaman (`BASE_KEYS` + `L3_KEYS` + `COMMON_VAR_KEYS`)

| Anahtar | Ne için |
|---|---|
| `I3EventHeader` | indeks + livetime (MJD alanları) |
| `IC2018_LE_L3_Vars` | **BDT'nin 4 muon + 2 noise girdisi burada** (map → çok kolon) |
| `IC2018_LE_L3_bools` | L3 kesim bayrakları |
| `L3_oscNext_bool` | `IC2018_LE_L3_Full AND Data_quality_bool` |
| `Data_quality_bool` | SLOP / LID errata veri kalitesi |
| `SRTTWSplitInIcePulsesDCHitStatistics` | `cog_z`, `z_sigma`, `z_travel` (muon BDT) |
| `SRTTWSplitInIcePulsesDCHitMultiplicity` | `n_hit_doms` (aday) |

### L4'te ürettiklerimiz (`L4_HDF5_KEYS`)

| Anahtar | Tip | BDT girdisi mi? |
|---|---|---|
| `L4_micro_count` | map | **evet** — noise |
| `L4_fill_ratio` | I3FillRatioInfo | **evet** — noise |
| `L4_iLineFit` + `L4_iLineFitParams` | I3Particle + Params | **evet** — noise (hız) |
| `L4_FullTimeLengthRatio` | I3Double | **evet** — noise |
| `L4_VICH_nch` | I3Double | **evet** — muon |
| `L4_accumulated_time` | I3Double | **evet** — muon |
| `L4_first_hlc_rho` | I3Double | **evet** — muon |
| `L4_first_hlc` | I3Particle | hayır (rho'nun kaynağı) |
| `L4_VICH_npulses`, `L4_VICH_qtot` | I3Double | hayır (tanı) |
| `L4_ToI` + `L4_ToIParams` | I3Particle + Params | hayır (aday/legacy) |
| `L4_separation_in_cogs` | I3Double | hayır |
| `L4_QR_Box` | — | hayır (slc-veto yok, üretilmiyor) |
| `L4_n_flux_events` | I3Double | hayır — **ağırlık** |
| `L4_Cut_Bool`, `L4_NoiseStraightCuts_Bool` | I3Bool | hayır |
| `L4_NoiseClassifier_ProbNu`, `L4_MuonClassifier_Data_ProbNu` | I3Double | `--apply-cut` ile |

### Örnek türüne göre (ağırlık için)

- `--mc --genie` → `I3MCWeightDict` (`OneWeight`, `NEvents`,
  `PrimaryNeutrinoEnergy/Zenith/Type`), `NEvPerFile`, `I3GenieSystWeightDict`
- `--noise` → `I3MCWeightDict`, `noise_weight` (**`MCInIcePrimary` YOK**)
- `--corsika` → `CorsikaWeightMap`, `PolyplopiaPrimary`, `I3CorsikaInfo`
- `--muongun` → `MuonWeight*` adaylarından hangisi varsa

### Sidecar: `<çıktı>.hdf5.meta.json`

`n_l3_files` (ağırlık böleni), `physics_frames`, `after_stream_filter`,
`booked`, `elapsed_s`. HDF5'in içine değil yanına yazılıyor.

---

# Bölüm 2 — Teknik notla karşılaştırma

## 2.1 Doğrulananlar

**Hiperparametreler (Tablo 10)** — `max_depth 6`, `num_leaves 25`,
`max_bin 32`, `min_data_in_leaf 500`, `feature_fraction` noise 0.8 / muon 0.7,
`lambda_l1 2.0`, `lambda_l2 1.0`, `min_gain_to_split 2.0`,
`is_unbalanced False`. `train_L4_classifier.py` ile birebir.

**Kesim değerleri** — noise `> 0.7`, muon `> 0.65` (§3.6.2, §3.6.3). Kodda aynı.

**Eğitim oranı** — noise BDT %33.3 train / %66.7 test (§3.6.2). Kodda aynı.

**Ön işleme** — "event weights manually re-scaled to balance the samples...
all weights re-scaled to be in range 0–1" (§3.6.1). `build_training_set`
tam olarak bunu yapıyor.

**`micro_count`** — Tablo 11: `L4_micro_count.STW_m3500p4000_DTW200`,
"[-3.5 µs, +4 µs], 200 ns kayan pencere, maks tetiklenen DOM sayısı".
Kodumuzdaki `MICROCOUNT_SUBKEY` **birebir aynı isim**, pencere ve genişlik
aynı. L3'ünki farklı: `STW9000_DTW300Hits` ([-4, +5] µs, 300 ns) — bu yüzden
ikisi tam korele değil, BDT ikisinden de bilgi çıkarıyor.

**`accumulated_time`** — Tablo 12: *"Time to reach 75% of an event's charge
in the cleaned pulse series."* Kodumuz: `fraction=0.75`, `pulses_key=cleaned_pulses`.
**Fraksiyon ve seri artık tahmin değil, doğrulanmış.** Açık kalan tek şey
referans zamanı: kodumuz `t[idx] − t[0]` (ilk pulse'a göre) alıyor; not
"time to reach" diyor ama sıfır noktasını söylemiyor. Tetikleme zamanına göre
de olabilirdi.

**`first_hlc_rho`** — Tablo 12: *"Radial distance from string 36 (roughly the
center of DeepCore) of the first HLC hit."* Kodumuz string 36 koordinatlarını
(`46.29, −34.88`) gömülü tutup aynı şeyi hesaplıyor.

**VICH hız penceresi ve veto DOM kümesi** — §3.4 (DeepCore Filter):

> *"the center-of-gravity (COG) of the hits inside the fiducial volume is
> calculated, then a relative velocity is derived between each hit of the veto
> region and the COG's position/time vertex. If the derived speed of any veto
> hit is contained within **[0.25, 0.4] m/ns**, the hit is discarded on the
> basis that the veto hit could be causally related to a muon crossing the
> detector."*

Bu, `VICH_SPEED = (0.25, 0.40)` değerimizin kaynağı — **doğrulandı**.
Ayrıca "veto region"un hangi tanım olduğu da netleşti: burada anlatılan
**DeepCore Filter'ın kendi fiducial/veto ayrımı**, yani
`icecube.DeepCore_Filter.DOMS`. Kodumuz tam onu kullanıyor — **doğru seçim**.

> Dikkat: L3'ün fiducial tanımı **farklı** (Tablo 7: DeepCore string 79–86
> DOM 11–60; IceCube string 25–27, 34–37, 44–47, 54 DOM 39–60). VICH için
> L3 tanımını kullanmak yanlış olurdu.

**`dt` yönü** — Not doğrudan söylemiyor ama fizik açık: muon detektörü
geçerken **önce** veto bölgesini vurur, sonra fiducial COG oluşur. Kodumuz
`dt = t_COG − t_hit > 0` şartını koyuyor — doğru yön.

---

## 2.2 Bulunan somut sapma: VICH'in COG'u

**Not (§3.4):** COG, **fiducial hacim içindeki** hitlerden hesaplanıyor
("the COG of the hits **inside the fiducial volume**").

**Kodumuz** (`oscNext_L4_variables.py`, `_vich`):

```python
cog = charge_weighted_cog(iter_hits(cln, geo))   # TUM temizlenmis seri
```

Yani veto bölgesindeki hitler de COG'a katılıyor. Muonlu olaylarda veto
hitleri COG'u **yukarı ve dışa** çeker → veto DOM'una olan mesafe `d` ve
`t_COG` değişir → hesaplanan hız kayar → `[0.25, 0.4]` penceresine düşme
olasılığı değişir. Etki muonlu olaylarda sistematik, nötrinolarda küçük;
yani **tam da ayırt etme gücünü bozacak yönde**.

**Düzeltme:** COG'u fiducial DOM'larla sınırla —
`deepcore_doms("IC86").DeepCoreFiducialDOMs`. Bu, kodda zaten hazır olan
listenin diğer yarısı.

Bu hâlâ notun tarifiyle *tam* eşleşmeyecek olabilir (not COG'un yük ağırlıklı
olup olmadığını söylemiyor; biz yük ağırlıklı alıyoruz), ama fiducial
sınırlaması notta **açıkça yazan** bir şey ve şu an kodda yok.

---

## 2.3 Kalan sapmalar

| Konu | Not | Bizde | Etki |
|---|---|---|---|
| **Muon BDT arka planı** | **gerçek dedektör verisi** (bu aşamada %99 muon), 2012–2017'den 18 run, mevsimsel muon akısı dengeli | CORSIKA | Not diyor ki MuonGun ile eğitilen sınıflandırıcı "benzer performans" verdi ama kullanılmadı. CORSIKA için de benzer beklenebilir; ama **data/MC kontrolü yapılamıyor** ve muon bundle gibi simüle edilmeyen popülasyonlar öğrenilemiyor. |
| **VICH pulse serisi** | DC Filter kendi SRT temizlemesini `SplitUncleanedInIcePulses` üzerinde yapıyor (tüm HLC hitleri korunuyor) | ham `SplitInIcePulses` | Yön belirsiz. Bizimki daha fazla hit görüyor → VICH biraz yüksek çıkabilir. |
| **`FullTimeLengthRatio`** | Tablo 11'de `IC2018_LE_L3_Vars.FullTimeLengthRatio` — yani **L3 değişkeni** | L4'te kendimiz hesaplıyoruz (pass3 L3 map'inde oran yok) | Bileşenler L3'te var (`CleanedFullTimeLength`, `UncleanedFullTimeLength` — Tablo 8). Not oranın yönünü söylemiyor; ama temizlenmiş seri temizlenmemişin alt kümesi olduğu için `cleaned/uncleaned ∈ [0,1]` sınırlı ve fiziksel — kodumuzun aldığı yön bu. |
| **HitStatistics anahtarı** | `SRTTWOfflinePulsesDCHitStatistics` (pass2) | `SRTTWSplitInIcePulsesDCHitStatistics` (pass3) | Bilinen pass2→pass3 yeniden adlandırma; `reference/pass3_L3_process.py` ile doğrulanmış. |
| **ντ** | Tablo 13'te ντ CC var (0.129 mHz) | set yok | Sinyalin ~%3'ü. |
| **BDT motoru** | LightGBM (§3.6.1) | pybdt (bilinçli sapma) | Tablo 10 parametreleri taşınamaz; `CLAUDE.md`. |

---

## 2.4 İsim uyuşmazlığı — kontrol edilmeli

Tablo 11 noise girdisini **`L4_iLineFit.speed`** diye veriyor — yani
`L4_iLineFit` **I3Particle**'ının `speed` alanı.

`oscNext_L4_pybdt.ipynb` REGISTRY'si ise:
```python
"iLineFit_speed": ("L4_iLineFitParams", "LFVel"),
```

İki ayrı sorun:

1. **Tablo vs Params.** `L4_iLineFit.speed` ile `L4_iLineFitParams.LFVel`
   normalde aynı değeri taşır (linefit fit hızını her ikisine de yazar),
   ama aynı olduğunu **doğrulamadan** güvenmeyin.
2. **Kolon adı.** LightGBM notebook'unda "DOĞRULANDI: pass3 hdfwriter kolonu
   `lf_vel`, eski varsayım `LFVel` DEĞİL" notu var. pybdt notebook'u hâlâ
   `LFVel` kullanıyor → kolon bulunamazsa **sessizce NaN** olur.

Elinde artık gerçek bir HDF5 var, 10 saniyede çözülür:

```python
# notebook bölüm 2
dump_tables(SAMPLES["nue"]["hdf5"].replace(".hdf5", "_smoke.hdf5"),
            only=["iLineFit"])
```

`check_registry` zaten bunu yakalamak için var — bölüm 3'ü çalıştırdığında
`[!] iLineFit_speed ... kolon yok` yazarsa sebep budur.

---

## 2.5 Ağırlık doğrulama hedefleri (Tablo 13)

L3'teki oranlar — ağırlık zincirinin **tek en iyi göstergesi**:

| Bileşen | L3 [mHz] | L4 noise sonrası | L4 tam | verim |
|---|---|---|---|---|
| GENIE νe CC | 0.95 | 0.90 | 0.84 | 88.5 % |
| GENIE νμ CC | 3.77 | 3.65 | 3.11 | 82.5 % |
| GENIE ντ CC | 0.129 | 0.124 | 0.119 | 92.1 % |
| GENIE ν NC | 0.53 | 0.51 | 0.46 | 85.5 % |
| Atm. μ (MuonGun) | 505 | 490 | 28.1 | 5.6 % |
| Vuvuzela gürültü | 36.6 | 0.28 | 0.28 | 0.7 % |
| **Toplam MC** | **547** | 495 | 32.9 | 6.0 % |

pass3'te birebir tutmaz, **mertebe** tutmalı.

> **Bu tablo bir hatayı ortaya çıkardı.** Notebook ağırlıkları
> `d["_n_files"]`'a bölüyordu ve bu **HDF5 dosya sayısı**ydı. 100 L3
> dosyası tek HDF5'e book edilince bölen 1 oluyordu → ağırlıklar **100 kat
> büyük**. νe için ~0.95 mHz yerine ~95 mHz görürdün. `process_L4.py` artık
> `<çıktı>.hdf5.meta.json` içine `n_l3_files` yazıyor, `load_sample` onu
> okuyor. Eski HDF5'lerin sidecar'ı yok → notebook uyarı basıyor.

---

# Öncelik sırası

1. **VICH COG'unu fiducial'la sınırla** (§2.2) — notta açıkça yazan, kodda
   olmayan tek şey. Muon ayrımını doğrudan etkiler.
2. **`iLineFit_speed` kolonunu doğrula** (§2.4) — 10 saniyelik iş, sessiz
   NaN riski.
3. **Ağırlıkları Tablo 13 ile karşılaştır** (§2.5) — `n_l3_files` düzeltmesi
   sonrası mertebe tutuyor mu?
4. `accumulated_time` referans zamanı (§2.1) — fraksiyon ve seri doğrulandı,
   sıfır noktası açık.
