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
- `icecube.hdfwriter` **cvmfs metaproject'inde** yoktu → `simple_booker.py`
  pytables ile fallback booking yapıyor. **Güncelleme:** kullanıcının kendi
  build'inde (`/data/user/$USER/icetray_build/build`) hdfwriter VAR —
  gerçek çalışmada `Booking: icecube.hdfwriter` yazıyor. Fallback artık
  devrede değil ama kod duruyor (cvmfs ortamına dönülürse gerekli).
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
oscNext_L4_pybdt.ipynb   ◄── ANA ARAYÜZ (aktif yol, 11 bölüm)
   0  Konfigürasyon + ortam kontrolü
   1  L3 → L4 işleme (process_L4.py'yi çağırır, +smoke test)
   2  Booking doğrulaması (HDF5'te gerçekte ne var)
   3  Feature registry (BDT değişkeni → HDF5 tablo/kolon)
   4  HDF5 → numpy (pytables; pandas YOK)
   5  Ağırlıklar (w_phys + eğitim ağırlığı)
   6  pybdt DataSet + train/test → .ds
   7  Eğitim (pybdt_train.py'yi çağırır)
   8  Doğrulama (Validator: KS overtraining, dist, rate)
   9  Kesim seçimi
  10  Frame'e uygulama + REGISTRY↔FEATURE_MAP tutarlılık kontrolü
   │
   ▼
pybdt_train.py  →  L4_{name}.bdt + .validator + .json + grafikler
   │
   ▼
pybdt_classifier_module.py  (PyBDTClassifier tray modülü)
   FEATURE_MAP'i l4_classifier_module.py'den alır
```

**Referans (LightGBM) yolu** — resmi yöntem, artık aktif kullanılmıyor:
`oscNext_L4_feature_engineering.ipynb` (eski arayüz, parquet export) →
`reference/train_L4_classifier.py` → `L4_{tag}_model.txt`/`.json` →
`l4_classifier_module.py`. Eski notebook'ta yeni notebook'a
**taşınmamış** bölümler var ve bu yüzden duruyor: §6 yeniden yazılan
değişkenlerin referansla karşılaştırılması, §8 türetilmiş değişkenler,
§9 data/MC uyumu, §10 korelasyon + incremental feature scan.

Yardımcı/tanı scriptleri:
- `icetray_env.py` — **tüm `icecube` import'ları buradan geçer**; import
  başarısız olursa sebebini raporlar. `require_pybdt()` de burada.
- `setup_env.sh` — ortamı bul / shell aç / tek komut çalıştır / Jupyter kernel.
- `scan_files.py` — bozuk `.i3.zst` dosyalarını bul, sağlam liste üret.
- `l4_run.py` — `process_L4.py` sürücüsü + canlı ilerleme çubuğu
  (`configure_runner`, `run_process`, `run_all`).
- `l4_data.py` — `REGISTRY`/`ALTS`, `dump_tables`, `check_registry`,
  `check_feature_map`, `load_sample`, `add_weights`.
- `AKIS_SEMASI.md` — **hangi dosya ne zaman çalışır**: iki aşamanın tam
  akışı, tray modül sırası, çalışmayan (referans) dosyalar.
- `TEKNIK_NOT_KARSILASTIRMA.md` — HDF5'e tam olarak ne yazdığımız +
  teknik notla satır satır karşılaştırma (Tablo 7/10/11/12/13).
- `diagnose_env.py` — ortamda ne var/yok (pybdt kontrolü dahil).

HDF5 kolonlarını dökmek için notebook bölüm 2.

## Kritik senkronizasyon noktası

`FEATURE_MAP` (`l4_classifier_module.py`) ile `REGISTRY` (`l4_data.py`)
birbiriyle satır satır aynı olmalı. Biri diğerinden farklı bir kolon
okursa model **sessizce** yanlış tahmin üretir — hata fırlatmaz.

Artık elle değil, kodla kontrol ediliyor: `l4_data.check_feature_map()`
`FEATURE_MAP`'i **AST ile** okuyor (icetray gerekmiyor) ve çakışmaları
listeliyor. FEATURE_MAP'in fazladan aday değişken içermesi normal;
tehlikeli olan **aynı isim, farklı kolon**.

İlk çalıştırmada gerçek bir çakışma yakaladı: `iLineFit_speed` REGISTRY'de
`LFVel`, FEATURE_MAP'te `lf_vel`, teknik not Tablo 11'de ise
`L4_iLineFit.speed`. Üçü de `l4_data.ALTS` içinde — dosyada **gerçekten
hangisi varsa** o kullanılıyor.

Model formatı bilinçli olarak joblib/pickle değil, LightGBM native metin
formatı (`.txt`) + JSON sidecar: IceTray ortamında sklearn/joblib yok, sadece
`lightgbm` + `numpy` var.

## Mevcut durum (README'den)

- [x] Ortam doğrulandı (`oscNext` projesi yok; `slc-veto` yok;
      kendi build'de `hdfwriter` VAR)
- [x] IceTray/pybdt import katmanı (`icetray_env.py` + `setup_env.sh`)
- [x] Bozuk girdi dosyalarına dayanıklılık (`--scan` + `--retries`)
- [x] νe işleme çalıştı (100 dosya → 256 799 olay, 1019 s)
- [x] CORSIKA işleme çalıştı (500 dosya → 6 462 olay, 2578 s)
- [ ] νμ / noise yeniden çalıştırılmalı — bozuk `.i3.zst` yüzünden yarım kaldı
- [x] Sütun isimleri kesinleştirildi (14/14 BDT girdisi bulundu)
- [ ] Yeniden yazılan değişkenler (VICH, accumulated_time, separation_in_cogs)
      referansla doğrulanmadı
- [ ] Ağırlıklar doğrulanmadı
- [ ] Model henüz eğitilmedi

## Açık riskler / fikir yürütülebilecek noktalar

> Teknik not okundu ve kodla satır satır karşılaştırıldı:
> **`TEKNIK_NOT_KARSILASTIRMA.md`**. Aşağıdaki 1–3 o karşılaştırmaya göre
> güncellendi.

1. **VICH** (`_vich`, `oscNext_L4_variables.py`) — büyük ölçüde **doğrulandı**:
   §3.4 hız penceresini ([0.25, 0.4] m/ns) ve "veto region"un DeepCore
   **Filter'ın** (L2) tanımı olduğunu açıkça yazıyor — `DeepCore_Filter.DOMS`
   kullanımımız doğru. (L3'ün Tablo 7'deki fiducial tanımı **farklı**, onu
   kullanmak yanlış olurdu.) `dt = t_COG − t_hit > 0` yönü de fizikle uyumlu.
   **Bulunan sapma (düzeltildi):** not COG'un *fiducial hacimdeki* hitlerden
   hesaplandığını söylüyor; kod tüm temizlenmiş seriyi kullanıyordu. Muonlu
   olaylarda veto hitleri COG'u yukarı çekiyordu (test: z −400 → −43).
   `fiducial_cog=True` artık varsayılan; `False` eski davranış.
   **Kalan açık:** COG yük ağırlıklı mı? Not söylemiyor, biz yük ağırlıklı
   alıyoruz. **Pulse serisi DOĞRULANDI:** orijinal pass2 kodu
   (`reference/oscNext_L4_pass2_original.py`) `I3CutL7Module`'e
   `InputPulses=uncleaned_pulses  # Use uncleaned hits` veriyor — bizim
   ham `SplitInIcePulses` kullanmamız orijinalle aynı.
2. **accumulated_time** — **doğrulandı**: Tablo 12 "Time to reach 75% of an
   event's charge in the cleaned pulse series" diyor; kod `fraction=0.75` ve
   `cleaned_pulses` kullanıyor. Fraksiyon ve seri artık tahmin değil.
   Orijinal pass2 kodu da Dunkman `CalculateVariables`'a
   `PulseSeries=cleaned_pulses` veriyor — seri seçimi **doğrulandı**.
   **Kalan açık:** referans zamanı — kod `t[idx] − t[0]` (ilk pulse) alıyor,
   not sıfır noktasını söylemiyor (tetikleme zamanı da olabilirdi); orijinalin
   içi de `analysis.event_selection` C++ kodunda, elimizde yok.
   `separation_in_cogs` BDT girdisi değil, düşük öncelik.
3. **FullTimeLengthRatio yönü — ÇÖZÜLDÜ.** Tablo 11 metni oranın yönünü
   söylemiyor ama **Şekil 13** söylüyor: `IC2018_LE_L3_Vars.FullTimeLengthRatio`
   dağılımının x ekseni **0.0 – 1.0** aralığında. Ters yön (uncleaned/cleaned)
   ≥ 1 olurdu ve bu eksene sığmazdı. Yani `cleaned / uncleaned` — kodun
   (`_full_time_length_ratio`) aldığı yön. Artık çıkarım değil, notta var.
   **Kalan (küçük) fark:** not bunu **L3 değişkeni** olarak listeliyor
   (`IC2018_LE_L3_Vars.FullTimeLengthRatio`); pass3 L3 map'inde oran yok,
   bileşenleri var (`CleanedFullTimeLength`, `UncleanedFullTimeLength`) —
   biz oranı L4'te bölerek üretiyoruz, değer aynı olmalı. **Doğrulandı:**
   `test_noise_vars.py` bölmeyi L3 bileşenleriyle karşılaştırıyor,
   143 olayda maks sapma 0.
   **Fizik açıklaması DÜZELTİLDİ (ölçümle).** Docstring "gerçek olay ~1,
   gürültü ~0" diyordu; gerçek ölçüm (126 νe + 17 noise):

   |  | oran (medyan) | temizlenmiş | temizlenmemiş |
   |---|---|---|---|
   | νe | 0.16 | 1626 ns | 10 100 ns |
   | noise (L3 geçen) | 0.27 | 2780 ns | 10 290 ns |

   Oran **hiçbir zaman 1'e yaklaşmıyor**: `SplitInIcePulses` tüm okuma
   penceresini kapladığı için temizlenmemiş süre her olayda ~10 µs, yani
   değişken fiilen "temizlenmiş süre / 10 µs". Ayırt etme yönü de **ters**:
   gürültünün temizlenmiş serisi νe'ninkinden daha uzun. Ayırt ediyor ama
   beklenen hikâye değil. (17 noise olayı az; sıralama tek dosyada böyle,
   magnitüd tespiti yapısal.)
   Kodda ayrıca `inf`/`NaN` koruması eklendi: `uncleaned <= 0` kontrolü
   sıfıra bölmeyi engelliyordu ama **NaN paydayı geçiriyordu**
   (`NaN <= 0` → False), sonuç sessizce NaN yazılıyordu. Artık sonuç
   `np.isfinite` ile denetleniyor ve oran > 1 çıkarsa bir kez uyarı basılıyor.
3a. **Muon BDT'de eksik girdi (düzeltildi).** Tablo 12 **10** değişken
   listeliyor, `MUON_FEATURES`'ta **9** vardı — `NchCleaned` atlanmıştı
   (noise listesinde olduğu için gözden kaçmış). Eklendi. Benzersiz BDT
   değişkeni: 14 (5 noise + 10 muon, `NchCleaned` ortak).
3c. **`noise_weight` kolonu (düzeltildi).** `AUX`'ta `("noise_weight",
   "value")` yazıyordu; eski LightGBM notebook'unda doğrulanmış hali
   `("noise_weight", "weight")`. Yanlış olduğu için gürültü ağırlığı
   **tamamen NaN** kalıyordu → noise örneğinin `w_phys`'i sıfır olurdu.
   Düzeltildi, ikisi de `ALTS`'te.
3b. **Kolon adları — ÇÖZÜLDÜ.** Gerçek pass3 çıktısı üzerinde
   `check_registry` ile doğrulandı: `fill_ratio` → **`fillratio_from_mean`**
   (alt çizgisiz), `iLineFit_speed` → `lf_vel`, `noise_weight` → `weight`.
   Üçü de `ALTS`'te. `bulundu: 5/5` ve `10/10` — 14 BDT girdisinin hepsi
   bulundu.
   HDF5 kolon adı ile frame alan adı aynı olmak zorunda değil (hdfwriter
   çeviricisi yeniden adlandırıyor); `check_feature_map()` iki tarafın da
   alternatiflerini dikkate alıyor.
3d. **CORSIKA ağırlığı (düzeltildi).** pybdt yolundaki `corsika_weight`
   her olaya **eşit** ağırlık veriyordu (`np.ones/n_files`) — yani
   ağırlıklandırma hiç yoktu. Sadece mutlak oranı bozmuyordu; muon BDT'si
   atmosferik spektrumu değil simülasyonun düz spektrumunu görüyordu,
   yani **eğitim setinin şekli yanlıştı**. Eski LightGBM notebook'undaki
   yöntem taşındı: `simweights` + `GaisserH3a`, yoksa
   `CorsikaWeightMap.Weight / (NEvents × OverSampling)`.
   Eşleştirme `Run/Event/SubEvent` üzerinden (satır sırasına güvenilmiyor).
   **Normalizasyon eski koddan bilerek farklı:** eski kod her HDF5 için
   `nfiles=1` verip sonda HDF5 sayısına bölüyordu; bizim parçalarımızda
   `--chunk-files` yüzünden birden fazla L3 dosyası var, o yüzden parçanın
   kendi `n_l3_files`'i `nfiles` olarak veriliyor ve sonda ayrıca bölme
   yapılmıyor.
4. **Ağırlık zinciri** (`PropagateGenieInfo`, `process_L4.py` MC_KEYS/
   NOISE_MC_KEYS/CORSIKA_KEYS) — pass3'te I3GenieInfo yoksa NEvents*fraksiyon
   fallback'ine düşülüyor; bunun ne sıklıkla tetiklendiği ve ne kadar sapma
   yarattığı ölçülmedi.
5. **fill_ratio — ÇÖZÜLDÜ (orijinal kodla).** Tablo 11 metni vertex'i
   *"about some vertex (details here)"* diye **boş bırakıyor** (doldurulmamış
   çapraz referans), yani nottan çıkarılamazdı. Orijinal pass2 kodu söylüyor:
   ana segment `oscNext_L4_noise_cut_variables`'a
   `fill_ratio_vertex=L4_FIRST_HLC_KEY` veriyor — yani **ilk HLC hit'in
   konumu**, bizim verdiğimizle aynı. `RecoPulseName=cleaned_pulses` ve
   `SphericalRadiusMean=1.6` de birebir aynı; orijinalin kendi yorumu:
   *"Was optimised for GRECO but has not been re-optimised for oscNext"* —
   yani 1.6'nın oscNext için ayarlanmamış olması **bilinen** bir durum,
   bizim eksiğimiz değil. Bölüm 10'daki feature importance / incremental
   scan ile yeniden optimize edilebilir (orijinal de bunu öneriyor).
5a. **iLineFit_speed — ÇÖZÜLDÜ (orijinal kodla).** Tablo 11 sadece
   *"Speed fitted by the improved LineFit algorithm"* diyor, parametre
   vermiyor. Orijinal pass2 kodu birebir bizim çağrımız:
   `tray.AddSegment(linefit.simple, ..., inputResponse=cleaned_pulses,
   fitName=L4_LINEFIT_KEY)`. Yani "improved LineFit" = `linefit.simple`,
   ekstra parametre yok. Şekil 13'ün x ekseni log ölçekte 10⁻³ – 10³ (m/ns).
5b. **micro_count: orijinal kod ile teknik not çelişiyor — KARAR: not.**
   Orijinal pass2 kodu zincire `uncleaned_pulses` ile başlıyor
   (`I3StaticTWC(InputResponse=uncleaned_pulses)`), Tablo 11 ise
   *"Start with the cleaned pulse series"* diyor. **Notu izliyoruz**
   (varsayılan, kalıcı karar).
   **Ölçüldü — fark neredeyse yok:** `test_noise_vars.py` iki zinciri aynı
   olayda hesaplıyor. νe'de 126 olayın 115'i, noise'da 17 olayın 16'sı
   **birebir eşit**; medyanlar aynı (5 ve 3). Sebep: zincirin sonundaki
   200 ns'lik pencere zaten belirleyici — gürültü hitleri ~10 µs'ye yayılmış
   olduğu için en yoğun 200 ns penceresine iki seride de aynı hitler düşüyor.
   Pass2'deki ölü SeededRT kodunun yıllarca fark edilmemesinin sebebi de bu.
   Yani düzeltme **doğru ama etkisi küçük**; "sınıflandırıcının ayırt etme
   gücü ciddi düşüyordu" değerlendirmesi (md. 4) fazla iddialıydı.
   Karşılaştırma için ikisi de üretilebilir:
   `process_L4.py --micro-count-uncleaned`, notebook'tan
   `run_all(extra_args=["--micro-count-uncleaned"])`, doğrudan segmentte
   `oscNext_L4(micro_count_uncleaned=True)`. `fill_ratio` her iki durumda da
   temizlenmiş seriyi kullanır (orijinalde de öyle).
   Ayrıntı için "Booking/okuma denetimi" md. 4.
   Orijinalden **doğrulanan** kısımlar: DeepCore fiducial `I3OMSelection`
   adımı (notta yok ama orijinalde var — bizde de var), `TriggerConfigIDs
   =[1010, 1011]`, `WindowMinus/Plus = 3500/4000`, `dtw = 200`, alt anahtar
   adı `STW_m%ip%i_DTW%i`, ve sayımın **DOM** sayısı olduğu
   (`len(reco_pulse_series.values())` — bizde `len(pmap)`, aynı şey).
5c. **YAPILACAK — sinyal train/test ayrımı iki BDT'de farklı.** Notebook
   bölüm 6'da `make_datasets` noise ve muon BDT'si için ayrı ayrı
   çağrılıyor ve her çağrıda `istrain = rng.random(...)` **yeniden**
   çekiliyor. İkisi de aynı νe+νμ sinyal olaylarını kullandığı için bir
   olay noise BDT'sinin *eğitim*, muon BDT'sinin *test* setinde olabiliyor.
   Tek tek modeller için sorun değil; ama L4 kesimi ikisinin **birleşimi**
   (orijinal: `noise ≥ 0.7 AND muon ≥ 0.65`) ve birleşik kesimi
   değerlendirecek ortak held-out set yok → nihai verim olduğundan iyi
   görünür. **Çözüm:** sinyal ayrımını bir kez çekip iki BDT'de de aynısını
   kullanmak (üç satır). Bilinçli olarak ertelendi.
5d. **`NOISE_NS_SCALE = 1e9` — DOĞRULANDI (ölçümle).** Vuvuzela
   `noise_weight`'inin birimi pass3'te 1/ns varsayılıyordu. İlk gerçek
   koşuda test setinin toplam gürültü oranı **20.5 mHz** çıktı; test seti
   2 051 olayın 1 016'sı (%49.5) olduğuna göre tam örnek **41.4 mHz** →
   teknik not Tablo 13'teki **36.6 mHz** ile **%13 uyum**. Varsayım doğru.
   Sinyal tarafı: 3.99 mHz (test) → 7.97 mHz (tam), notun νe CC + νμ CC +
   NC toplamı 5.25 mHz — **1.5 kat**, uydurma E⁻³ güç yasası için beklenen.
   Ayrıca gürültü ağırlıkları **tam olarak düzgün** (her olay %0.10 =
   1/1016): vuvuzela sabit livetime üzerinde üretildiği için beklenen,
   hata değil. Sonuç: gürültü için ağırlıklı ve ağırlıksız verim/red
   **her kesimde birebir aynı**.
5e. **Noise MC istatistiği — ÖLÇÜLDÜ, YETERSİZ.** 100 L3 dosyasından
   toplam **2 051** olay (train 1 035 / test 1 016); sinyal 329 545.
   Oran **159:1**. `test_noise_vars.py`'deki %8'lik L3 geçme oranıyla
   (νe'de %84) tutarlı. Referansın istatistiği çok daha büyük olmalı:
   Tablo 10 noise BDT'si için `min data in leaf = 500` veriyor — bizim
   1 035 eğitim olayımızla bu ayar 2 yaprak demek olurdu.
5f. **İlk eğitilen noise BDT — çalışıyor ama referanstan zayıf.**
   pybdt/AdaBoost, 300 ağaç, derinlik 3, `p_KS` sinyal 0.207 / arkaplan
   0.099 (**overtraining YOK** — notun kendi noise BDT'si için
   *"some overtraining is observed"* dediği düşünülürse iyi).
   Verim/red eğrisi (test, `w_phys`):

   | kesim | red | sinyal verimi |
   |---|---|---|
   | 0.00 | %34 | %99.1 |
   | 0.50 | %67 | %96.1 |
   | 0.70 | %81 | %90.5 |
   | 0.90 | %95 | %78.1 |
   | **0.945** | **%98.9** | **%52.7** |

   **Hedef (Tablo 13): %99.2 redde ~%96 verim.** Aradaki fark ağırlıktan
   DEĞİL (yukarıda elendi), iki şeyden: (a) arka plan istatistiği (5e),
   (b) model kapasitesi — Tablo 10 `max depth = 6`, `num leaves = 25`
   diyor; bizim `--depth 3` (≤8 yaprak) keyfi bir seçimdi, nottan
   gelmiyor. `--prune-strength 35` de keyfi ve fazla agresif olabilir.
   Sıradaki denemeler: daha çok noise dosyası; `--depth 6`,
   `--num-trees 500`, budamasız.
6. **ντ ve gerçek dedektör verisi yok** — sinyal tanımı νe+νμ (ντ CC ~%3),
   muon BDT arka planı CORSIKA (gerçek veri değil). Bu ikame ne kadar
   sapma yaratıyor, data/MC uyum kontrolü (bölüm 9) devreye girince
   netleşecek.

## Booking/okuma denetimi — bulunan hatalar

Kod baştan sona gözden geçirildi; üçü **sessiz veri bozulması** üreten
gerçek hatalardı.

**1. `__I3Index__` gerçek veriyi eziyordu (düzeltildi).**
`h5.walk_nodes("/", "Table")` alt gruplara da iniyor. hdfwriter her anahtar
için `/X` (veri) ve `/__I3Index__/X` (indeks) yazıyor; sözlük leaf isimle
kurulunca indeks veriyi eziyordu → her tablo `start/stop` kolonlu görünüyor,
**tüm değişkenler NaN** oluyordu. `_table_nodes()` `__I3Index__` altını
atlıyor.

**2. Tekrar eden `Run/Event/SubEvent` yanlış eşleştirme (düzeltildi).**
Bir tablo `I3EventHeader` ile hizalı değilse (anahtar bazı frame'lerde
yoksa) eşleştirme `(Run, Event, SubEvent)` sözlüğüyle yapılıyordu. Bu üçlü
**benzersiz değil**: MC'de `run_id` = set no, `event_id` her L3 dosyasında
sıfırdan başlıyor ve `--chunk-files` ile bir parçada 10 dosya var. Sözlükte
son gelen kazanıyor → olaylar ya NaN kalıyor ya **başka bir olayın
değerini** alıyordu.
Düzeltme: eşleştirme artık hdfwriter'ın `/__I3Index__/<anahtar>`
tablosundan (`exists`/`start`) yapılıyor. Indeks yoksa ve üçlüler
tekrarlıyorsa sessizce yanlış eşleştirmek yerine NaN bırakılıp bildiriliyor.

**3. `n_flux_events` dosya başına güncellenmiyordu (düzeltildi).**
`PropagateGenieInfo` değeri bir kez okuyup sabitliyordu. Bir tray
`--chunk-files` ile 10 L3 dosyası işliyor ve **her dosyanın kendi
`I3GenieInfo`'su** var → ilk dosyanın değeri hepsine uygulanıyordu, sonraki
dosyaların ağırlıkları yanlış çıkıyordu. Artık her `I3GenieInfo`'da
güncelleniyor, değer değişirse loglanıyor.

**4. `micro_count` gürültü temizliği olmadan sayılıyordu (düzeltildi).**
Zincir `uncleaned_pulses` → `I3StaticTWC` → `I3SeededRTCleaning` →
`I3OMSelection` → `I3TimeWindowCleaning` → say olarak **belgelenmişti**, ama
`I3OMSelection` girdi olarak SeededRT çıktısını (`L4_SRTTWPulses`) değil
StaticTWC çıktısını (`L4_TWPulses`) alıyordu. `L4_SRTTWPulses` frame'e
yazılıp **hiçbir yerde okunmuyordu** (grep ile doğrulandı) — yani zincirdeki
tek gürültü temizleme adımı fiilen devre dışıydı.

**Bu hata bizim değil, orijinal pass2 kodundan miras** — sonradan bulundu:
`reference/oscNext_L4_pass2_original.py` içinde aynı satırlar duruyor ve
yazarın kendi yorumu bile şüpheli: `srt_tw_pulses = "L4_SRTTWPulses"
#TODO Is this actually used?`. Yani **pass2 sayıları da gürültü temizliği
olmadan üretilmiş** (Şekil 12'nin micro_count paneli bu haliyle).
Bizim düzeltmemiz **notu** izliyor, orijinal kodu değil — bilinçli sapma.

Sonuç: `micro_count` ham hitler üzerinden sayılıyordu. Kalan adımların hiçbiri
gürültü elemiyor (StaticTWC geniş zaman kesiti, OMSelection uzaysal kesim,
TimeWindowCleaning en yoğun 200 ns penceresi).

**Etkisi sonradan ölçüldü ve KÜÇÜK çıktı** (bkz. açık risk 5b): iki zincir
νe'de olayların %91'inde, noise'da %94'ünde birebir aynı sayıyı veriyor.
200 ns'lik pencere zaten gürültüyü fiilen eliyor. Düzeltme nota uygunluk
için doğru, ama buraya ilk yazılan "ayırt etme gücü düşüyordu" ifadesi
ölçümle desteklenmiyor — abartılıydı.

Teknik not (Tablo 11) *"Start with the **cleaned** pulse series"* diyor.
Düzeltme nota göre yapıldı: zincir artık `cleaned_pulses`
(`SRTTWSplitInIcePulsesDC`) ile başlıyor ve SeededRT bloğu kaldırıldı —
L3 o seriye zaten SRT temizliği uygulamış (adındaki "SRT" bu), tekrarı
çift temizleme olurdu. Yeni zincir notun dört adımıyla birebir:
`cleaned → StaticTWC [-3500,+4000] ns → DeepCore fiducial → 200 ns DTW → say`.

`oscNext_L4_noise_cut_variables` artık `uncleaned_pulses` parametresi
almıyor; segmentte STTools bağımlılığı da kalmadı.

**Bu düzeltme mevcut HDF5'leri geçersiz kılar** — νe ve CORSIKA yeniden
işlenmeli (νμ/noise zaten yeniden işlenecekti).

**Küçük:** `--n` ile üretilen smoke çıktısında `n_l3_files` yanıltıcıydı
(tray erken duruyor, liste tamamen okunmuyor). Artık `None` +
`n_l3_files_unreliable` yazılıyor, `load_sample` bölen olarak kullanmıyor.
`PropagateGenieInfo`'daki `DAQ()`/`Simulation()` ölü koddu (`Process()`
override edilince çağrılmıyorlar) — kaldırıldı.

## Referans belgeler (`reference/`)

- `reference/OscNext_v00.074_pass2_technical_note.pdf` — pass2 için resmi
  oscNext teknik notu (83 sayfa). **pass3 için değil**, ama L4 mantığının
  büyük kısmı (değişken tanımları, BDT hiperparametreleri) pass3'te de
  aynı kabul ediliyor.
- `reference/oscNext_L4_pass2_original.py` — **orijinal oscNext L4 tray
  segment'i** (Tom Stuttard, pass2). Tüm gövde yorum satırı hâlinde
  (`#TODO migrate` — GitHub IceTray'e taşınmamış), ama parametre değerleri
  ve modül zincirleri **birinci elden kaynak**. Bizim
  `oscNext_L4_variables.py`'miz bunun yeniden yazımı. Doğruladığı şeyler:
  `fill_ratio_vertex=L4_FIRST_HLC_KEY`, `SphericalRadiusMean=1.6`,
  `linefit.simple`, VICH'in `uncleaned_pulses` kullanması, Dunkman'ın
  `cleaned_pulses` kullanması, micro_count parametreleri, straight-cut
  eşikleri, ve L4 kesim eşikleri (noise ProbNu ≥ 0.7, muon ProbNu ≥ 0.65).
  Tek çelişki: micro_count zincirinin `uncleaned_pulses` ile başlaması
  (bkz. açık risk 5b).
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

  **Dikkat:** Bu script `IC2018_LE_L3_Vars`'ı hiç anmıyor (ne yazıyor ne
  okuyor) — `DeepCoreCuts`'ın onu da ürettiği bir *çıkarımdı*. Aşağıdaki
  gerçek dosya dökümüyle doğrulandı.

## Gerçek L3 dosyasında ne var (doğrulandı)

`genie_NuE_IC86.023800.000000.i3.zst` Physics frame'i dökülerek
**doğrulandı** — artık varsayım değil:

- **`IC2018_LE_L3_Vars`** (`I3MapStringDouble`) **VAR**, 14 kolon:
  `C2HR6`, `CausalVetoHits`, `CleanedFullTimeLength`, `DCFiducialHits`,
  `ICVetoHits`, `NAbove200Hits`, `NchCleaned`, `NoiseEngine`,
  `RTVeto250Hits`, `RTVetoCutHit`, `STW9000_DTW300Hits`,
  `UncleanedFullTimeLength`, `VertexGuessZ`, `VetoFiducialRatioHits`.
  `FEATURE_MAP`/`REGISTRY`'de L3'ten okuduğumuz her kolon bu listede.
- `IC2018_LE_L3_bools` (`I3MapStringBool`): `IC2018_LE_L3_Full`,
  `..._No_Nch`, `..._No_Nch_No_RTVeto`, `..._No_RTVeto`,
  `..._No_UncleanedTime`
- `SRTTWSplitInIcePulsesDC` ve `SplitInIcePulses` VAR — ikisi de
  `I3RecoPulseSeriesMapMask`, `get_pulses()` bunları `.apply(frame)` ile
  açıyor
- `L3_oscNext_bool`, `Data_quality_bool` VAR
- `I3GenieInfo` VAR → ağırlıkta `NEvents × 0.7/0.3` fallback'ine
  düşmemeliyiz
- `MCInIcePrimary` **YOK** → truth bilgisi `I3MCWeightDict`'ten alınmalı
- HitStatistics / HitMultiplicity **YOK** → L3 bunları siliyor, bizim
  yeniden hesaplamamız (`oscNext_L4_hit_statistics`) gerekli

**İki tuzak:**
1. `I3GenieResult` deserialize **edilemiyor**: *"Attempting to read
   version 2 from file but running version 1 of I3GenieResult"* — dosya
   kurulu `simclasses`'tan yeni. Bu anahtarı book etmediğimiz için
   şimdilik zararsız, ama `--output-i3` kullanılırsa iş çökebilir.
2. Frame'de ayrıca `pole_grecofilter_onlineLowEnL3_Vars` var — bu
   *online* filtrenin ayrı map'i, `IC2018_LE_L3_Vars` ile karıştırılmamalı.

## BDT eğitimi: pybdt kullanılacak (bilinçli sapma — dikkat)

`icecube/icetray` (private) içindeki `pybdt` (IceCube'un kendi AdaBoost
tabanlı BDT kütüphanesi, C++/boost_python; kod repoda `pybdt/` altında)
bu projede **kasıtlı olarak** kullanılacak.

**Bunun resmi oscNext analiziyle uyuşmadığını bil:** Teknik not (pass2,
v00.074, bölüm 3.6.1) L4 noise/muon sınıflandırıcılarının LightGBM
(gradient boosting) ile eğitildiğini açıkça yazıyor, pybdt hiç
geçmiyor; Tablo 10'daki hiperparametreler de (`max_depth`, `num_leaves`,
`max_bin`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`) LightGBM'in
native isimleri. `reference/train_L4_classifier.py` bu resmi yaklaşımı (LightGBM)
uyguluyor ve referans olarak repoda duruyor — **ama üç kusuru var,
sayılarına güvenmeyin** (incelendi):

1. **IceTray ortamında import EDİLEMEZ.** Modül seviyesinde
   `import pandas` ve `from sklearn.metrics import ...` yapıyor —
   kendi docstring'i *"IceTray ortamında sklearn ve joblib YOK"*
   dediği hâlde. Tablo 10 değerlerine ihtiyaç duyan kod bu dosyayı
   import edemez; `lightgbm_compare.py` bu yüzden `PARAMS`'ı **AST ile
   okuyor** (aynı çözüm `l4_data.check_feature_map()`'te de var).
2. **Early stopping TEST setiyle yapılıyor** — `valid_sets=[dtrain,
   dtest]` ve durma kararı `dtest`'ten geliyor, sonra aynı test setinde
   AUC raporlanıyor. Test sızması; raporlanan sayı iyimser.
3. **Kesim performansı train+test KARIŞIK hesaplanıyor.** `evaluate()`
   içinde `wp[(y == 1) & (prob >= c)]` — `& te` maskesi YOK. Yani
   "sinyal verimi / arka plan reddi" satırı eğitim olaylarını da
   içeriyor. Tablo 13 ile karşılaştırılacak sayı tam olarak bu
   olduğu için önemli bir hata.

`lightgbm_compare.py` üçünden de kaçınıyor: import etmiyor (AST),
early stopping için eğitim setinden ayrılan dilimi kullanıyor, ve
yalnızca test setinde ölçüyor.

pybdt'ye geçiş şu sonuçları doğurur:
- pybdt AdaBoost yapıyor, LightGBM'in gradient-boosting + leaf/lambda
  regularizasyon mantığı yok — Tablo 10 parametreleri pybdt'ye
  **doğrudan taşınamaz**, pybdt'nin kendi API'sine göre (`num_trees`,
  `beta`, `depth`, `min_split`, `prune_strength`, `use_purity`) ayrıca
  ayarlanmalı. `pybdt_train.py` bunları CLI bayrağı olarak alır ve
  verilmeyenleri pybdt'nin kendi varsayılanlarında bırakır — oscNext
  için optimize edilmiş bir değer kümesi YOK.
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
  `reference/train_L4_classifier.py`'nin aksine pybdt `icecube.*` namespace
  paketi değil, bağımsız üst düzey bir pip-tarzı pakettir.
- Bu özel build'in ortamı: `eval $(/cvmfs/.../py3-v4.4.2/setup.sh)` →
  `cd <build_dizini> && ./env-shell.sh`. Jupyter de bu ortamdan
  başlatılmalı (Jupyter içindeki terminaller/kernel'ler ortamı miras
  alır, tekrar env-shell gerekmez; sadece Jupyter sunucusu yeniden
  başladığında bu iki adım tekrarlanır).
### pybdt yolu (yeni, LightGBM yolundan bağımsız)

pybdt'nin kendi önerdiği iş akışını izler (bkz. `pybdt/resources/docs/`
`man_training.rst`, `man_validator_setup.rst`):

```
.ds DataSet dosyaları  --BDTLearner-->  .bdt  --Validator-->  grafikler
```

- `pybdt_train.py` — eğitim + doğrulama, **tamamen standalone**:
  sklearn / lightgbm / pandas kullanmaz, `reference/train_L4_classifier.py`'den
  hiçbir şey import etmez. Girdi olarak pybdt native `.ds` dosyaları
  alır. Değerlendirme için pybdt'nin kendi `validate.Validator`'ını
  kullanır — KS testli overtraining kontrolü (`p_KS < 0.01` uyarısı,
  pybdt dokümantasyonunun eşiği), skor dağılımı ve rate-vs-cut
  grafikleri hazır gelir. Hiperparametreler yalnızca komut satırında
  açıkça verilirse set edilir, gerisi pybdt'nin kendi varsayılanlarında
  kalır; fiilen kullanılan değerler `.json` meta dosyasına yazılır.
- `pybdt_classifier_module.py` — `PyBDTClassifier` tray modülü,
  `.bdt` + `.json`'ı okuyup frame'e `I3Double` yazar. Frame'den değişken
  okuma için `l4_classifier_module.py`'deki `FEATURE_MAP`/`read_feature`
  import edilir (mapping tek yerde kalsın diye; o modül lightgbm'i
  sadece kendi `load_model()`'ı içinde import ettiği için bu bağımlılık
  lightgbm gerektirmez).

- `oscNext_L4_pybdt.ipynb` — **ana arayüz**. Uçtan uca tüm süreç: L3→L4
  işleme, booking doğrulaması, feature registry, HDF5→numpy, ağırlıklar,
  `.ds` üretimi, eğitim (`pybdt_train.py`'yi subprocess olarak çağırır),
  doğrulama, kesim seçimi. Yalnızca numpy + pytables + pybdt kullanır —
  **pandas kullanmaz** (IceTray ortamında bulunmayabilir). Ağırlık
  mantığı eski notebook'un 7. bölümünden taşındı.

Eğitim mantığı notebook'ta **tekrarlanmıyor**, `pybdt_train.py` subprocess
olarak çağrılıyor — tek implementasyon kalsın diye (eski notebook da
`process_L4.py`'yi böyle çağırıyordu).

**Notebook ince bir arayüz.** Ağır mantık `l4_run.py` + `l4_data.py`
içinde; notebook onları import ediyor. Notebook versiyonlanıyor (tracked)
ama çalıştırınca çıktı hücreleri `git pull`u bloklayabilir — bunu önlemek
için bir kez `pip install --user nbstripout && nbstripout --install`.

**Dikkat — `.ds` dosyalarında BDT girdisi olmayan kolonlar var** (`w_phys`,
fiziksel ağırlık). Notebook eğitime `--features`'ı **açıkça** geçer;
`pybdt_train.py` de `--features` verilmezse `RESERVED_COLS`'u dışarıda
bırakır. Bu koruma olmazsa model fiziksel ağırlığı bir değişken sanıp
öğrenir — sessiz ve ciddi bir hata.

**Henüz test edilmedi:** notebook uçtan uca hiç çalıştırılmadı; HDF5 sütun
isimleri doğrulanmadığı için 3. bölümdeki `REGISTRY` büyük olasılıkla
düzeltme gerektirecek (2. bölüm zaten bunu tespit etmek için var).

## Ortam kurulumu — bilinen tuzaklar

**1. `env-shell.sh` yeni bir shell açar.** Script içinde ard arda
`./env-shell.sh` ve `python ...` yazarsan ikinci satır o shell'den
çıkıldıktan sonra, yani ortam olmadan çalışır. Bu, "icetray import
edilmiyor" hatasının 1 numaralı sebebi. Tek komut için:
`./env-shell.sh -- python ...` ya da `./setup_env.sh run python ...`.

**2a. `cannot import name '...' from 'l4_data'` — modül önbelleği.**
Python bir modülü bir kez import edince kernel'de tutar; `git pull` dosyayı
güncelleşe bile `from l4_data import yeni_fonksiyon` eski modül nesnesine
bakar ve ImportError verir. Ayırt etmek için:
```python
import l4_data
print("dosyada  :", "def aux_for" in open(l4_data.__file__).read())
print("hafizada :", hasattr(l4_data, "aux_for"))
```
`dosyada True, hafizada False` → önbellek; **Kernel → Restart**.
İkisi de False → `git pull` yapılmamış.
Notebook bölüm 0 artık `%autoreload 2` açıyor, bu sorun tekrarlamamalı.

**2b. Jupyter'in CALISMA DIZINI kernel restart ile degismez.**
Belirti: alt surec `returncode=2` ile oluyor ve yol
`~/.local/share/Trash/files/...` gibi bir yeri gosteriyor. Sebep: repo
klasoru silinmis/tasinmis ama Jupyter sunucusu hala eski dizinde
calisiyor; `./process_L4.py` gibi **goreli** yollar oraya cozuluyor.
Ayni sebep eski `l4_data.py`'nin okunmasina da yol acar.

Kernel Restart YETMEZ — calisma dizini **sunucudan** gelir. Jupyter
sunucusunu kapatip dogru dizinden yeniden baslat:
```bash
cd ~/l4/osncnextl4 && python -m jupyter lab --no-browser --port=8896
```
Notebook'ta kontrol: `import os; os.getcwd()`.
`configure_runner` artik bunu basta yakalayip soyluyor.

> Klasoru dosya yoneticisinden ya da Jupyter'in sil dugmesinden silmek
> `rm` degil, **cop kutusuna tasima**dir (`~/.local/share/Trash/files/`).
> Dosyalar orada durur ve home kotasindan yer yer — kurtarilacak bir sey
> varsa oradan alin, sonra `rm -rf` ile gercekten silin.

**2. Jupyter kernel'i.** Notebook'un IceTray/pybdt'yi görmesinin tek yolu
kernel'in env-shell içindeki python olması. Jupyter'i ortam içinden
başlatmak en temizi (README adım 4); başlatılmadıysa `./setup_env.sh kernel`
ile kernel kaydedilip notebook'ta seçilir.

**3. `import pybdt`, `from icecube import pybdt` DEĞİL.** pybdt `icecube`
isim alanında değil, bağımsız üst düzey bir paket. Bu proje geçmişinde bir
kez "pybdt derlenmemiş" sanılmasına yol açtı. `icetray_env.require_pybdt()`
bu hatayı yakalayıp doğru kullanımı söylüyor.

**4. `I3Tray`'in yeri sürüme göre değişiyor** — `icecube.icetray.I3Tray`
(v1.5+) vs top-level `I3Tray` (combo). `icetray_env.get_I3Tray()` ikisini
de dener.

**5. Tek eksik proje tüm repoyu kilitliyordu.** `oscNext_L4_variables.py`
eskiden modül seviyesinde `tensor_of_inertia`, `fill_ratio`,
`DeepCore_Filter` import ediyordu; biri eksikse dosya hiç import
edilemiyordu. Artık `optional_project()` ile; eksik proje, o değişkeni
üreten segment çağrıldığında (`require_project(...)`) net hata veriyor.

Build arama sırası (`setup_env.sh` ve `icetray_env.find_env_shells()`):
`$OSCNEXT_I3_BUILD` → `$I3_BUILD` → `/data/user/$USER/icetray_build/build`
→ `/data/user/$USER/*/build` → `~/*/build` → cvmfs metaproject'leri.

## Bozuk girdi dosyaları (çözüldü)

pass3 üretiminde yarım yazılmış `.i3.zst` dosyaları var:

```
FATAL (I3Reader): Error reading .../genie_NuMu_IC86.023799.000046.i3.zst
                  at frame 4: input stream error!
```

`I3Reader` tüm dosya listesini tek seferde alıyor → bir dosya bozuksa **tüm
tray ölüyor**. numu (100 dosya) ve noise (100 dosya) job'ları bu yüzden
yarım kaldı. Ardından gelen `Table ... is still connected ... This is a
BUG!` mesajı **bunun sonucu**, ayrı bir bug değil — tray patlayınca HDF5
düzgün kapanmadı demek. O yarım HDF5'ler açılamaz, silinmeli.

`process_L4.py` iki katmanlı koruma yapıyor:
1. **Ön tarama** (`--scan quick`, varsayılan) — dosya başına ilk 25 frame
   okunur, açılmayanlar elenir.
2. **Çalışma anı** (`--retries 3`, varsayılan) — tray yine patlarsa hata
   mesajından dosya adı regex ile çıkarılır, kara listeye yazılır, yarım
   HDF5 silinir, o dosya hariç yeniden denenir. (Tray bir fabrika
   fonksiyonuna alındı: bir `I3Tray` ikinci kez `Execute` edilemiyor.)

Elenen dosyalar `<çıktı>.hdf5.badfiles.txt`'ye yazılır. Set başına bir kez
`scan_files.py --good-list` ile tarayıp `--input-list ... --scan off`
kullanmak en verimlisi.

## HDF5 üretimini hızlandırma

Ölçülen: νe 100 dosya / 1019 s (~10 s/dosya), CORSIKA 500 dosya / 2578 s.
Tek süreç, tek çekirdek.

**Önce ölç, sonra optimize et:**
```bash
python process_L4.py --usage --n 2000 --scan off --input <tek_dosya> ...
```
IceTray modül bazlı CPU zamanını basar. En yavaş modülü görmeden
optimize etmeye çalışma.

**Hızlandırma OTOMATİK DEĞİL** — açıkça istenmeli. Varsayılan
`run_all()` hâlâ tek süreç ve tüm değişkenleri hesaplar.

**1. Paralellik (en büyük kazanç).** Girdi dosyalarını N sürece böl:
```python
run_all(jobs=8, chunk_files=10, skip_optional=True)   # tüm örnekler
run_process_parallel("numu", jobs=8, chunk_files=10)  # tek örnek
```
Ayrı süreçler, ayrı çıktılar (`L4_nue_job0_part000.hdf5` …), ortak durum
yok → neredeyse doğrusal hızlanma. Hepsi `L4_nue*.hdf5` glob'una uyar ve
her parça kendi `.meta.json`'ını yazar, `n_l3_files` doğru toplanır.
**cobalt paylaşılan makine** — `jobs=8` makul, `jobs=64` değil.

**2. Gereksiz hesabı atla.** `--skip-optional` → `I3TensorOfInertia`
(`L4_ToI`) ve `separation_in_cogs` hesaplanmaz. İkisi de Tablo 11/12'de
**yok**, yani BDT girdisi değil; aday/legacy olarak duruyorlar.

**3. Taramayı bir kez yap.** `scan_files.py --good-list` ile set başına bir
kez tara, sonra `--input-list ... --scan off`. `run_process_parallel`
zaten `--scan off` kullanıyor.

**4. Az anahtar book et.** 33 anahtar yazılıyor; `I3GenieSystWeightDict`
gibi büyük map'ler gerekmiyorsa `process_L4.build_key_list`'ten çıkarmak
hem yazmayı hızlandırır hem dosyayı küçültür.

## `--n` frame sayar, olay saymaz

`process_L4.py --n N` → `tray.Execute(N)` → **N frame** işlenir. Frame ≠ olay:
akışta G/C/D (GCD), Q (DAQ) ve P (Physics) frame'leri var; bir DAQ olayı
birden fazla P frame (sub-event) üretebilir. Üstelik P frame'lerin ancak bir
kısmı `--sub-event-stream`'e uyar ve ancak bir kısmı L3 kesimini geçer.

Bu yüzden `--n 200` ile 60 olay book edilmesi normal. Çıktı artık kademeyi
gösteriyor:

```
Physics frame           : 98
  InIceSplit            : 98  (100.0%)
  L3 kesimi sonrasi     : 60  (61.2%)
Book edilen olay        : 60
```

Kayıp nerede olursa olsun burada görünür: `InIceSplit` satırı 0 ise
`--sub-event-stream` yanlış; L3 satırı 0 ise girdi L3 çıktısı değil ya da
`Data_quality_bool` eliyor (test için `--no-l3-cut`).

`--n` verildiğinde tray erken durur, yani **dosya listesinin tamamı okunmaz**.
Bu yüzden `--n` modunda "işlenen dosya" sayısı basılmıyor (yanıltıcı olurdu:
listede 100 dosya olsa da tray ilk dosyada durmuş olabilir). Smoke test'te
tek dosya verin ya da `--scan off` kullanın — 100 dosyayı taramak boşuna.

## Konvansiyonlar

- Kod ve yorumlar Türkçe.
- Veri repoya girmez (`.gitignore`: `L4_output/`, model/veri uzantıları).
- Notebook commit'lenmeden önce `nbstripout` ile temizlenmeli (çıktı hücreleri
  MB'larca yer kaplar ve anlamsız diff üretir).
- `oscNext_L4_variables.py` içindeki her yeniden yazılmış fonksiyonun
  docstring'inde orijinalin nereden geldiği ve neden değiştiği yazılı —
  değişiklik yapmadan önce bu docstring'leri okuyun.
