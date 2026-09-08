# Notebook rehberi

Bu branch'te **iki notebook** var:

| Notebook | Durum | Motor |
|---|---|---|
| `oscNext_L4_pybdt.ipynb` | **ANA ARAYÜZ** (aktif yol) | pybdt |
| `oscNext_L4_feature_engineering.ipynb` | referans / eski arayüz | LightGBM |

Aşağıdaki bölüm bölüm açıklama **`oscNext_L4_feature_engineering.ipynb`**
için. Eski notebook duruyor çünkü pybdt notebook'una **taşınmamış** bölümleri
var ve hâlâ gerekli:

- §6 yeniden yazılan değişkenlerin referansla karşılaştırılması (VICH!)
- §8 türetilmiş değişkenler
- §9 data/MC uyum kontrolü
- §10 korelasyon + incremental feature scan

Bölüm 0–5 ve 7 (ağırlıklar) pybdt notebook'una taşındı, mantık aynı — orada
pandas yerine saf numpy + pytables kullanılıyor. Yani aşağıdaki açıklamaların
büyük kısmı iki notebook için de geçerli.

Dosyanın sonunda **pybdt'nin nasıl çalıştığı** ayrı bir bölüm olarak var.

---

## `oscNext_L4_feature_engineering.ipynb` — adım adım

Notebook bir "analiz defteri" değil, **üretim hattının kontrol paneli**.
Ağır iş (`.i3` okuma, tray çalıştırma) IceTray'de `process_L4.py` ile yapılır;
notebook o çıktıyı okur, doğrular, değişken üretir ve BDT eğitimi için
parquet hazırlar.

```
L3 .i3.zst ──process_L4.py──▶ L4 .hdf5 ──notebook──▶ L4_*_training.parquet
                (IceTray)                (pandas)          │
                                                            ▼
                                              train_L4_classifier.py
                                                     (LightGBM)
                                                            │
                                                            ▼
                                       l4_classifier_module.py (IceTray'de uygula)
```

Notebook'un kendisi **IceTray'e ihtiyaç duymaz** (bölüm 1'den job başlatmazsan).
Sadece `numpy / pandas / tables / matplotlib / lightgbm` yeter. Bu bilinçli:
HDF5 zaten üretildikten sonra IceTray'e gerek kalmıyor.

> pybdt notebook'u bu konuda daha katı: **pandas hiç kullanmıyor** (IceTray
> ortamında bulunmayabilir), sadece numpy + pytables + pybdt.

---

## Bölüm 0 — Konfigürasyon

**Ne yapıyor:** yolları kuruyor ve `oscNext_L4_variables.py` içindeki frame
anahtar isimlerini **AST ile** okuyor.

`load_constants()` dosyayı `import` etmiyor, `ast.parse` ile ayrıştırıp modül
seviyesindeki atamaları çözüyor. `ast.Constant`, `ast.Name`, `+`, `%`, `*` ve
tuple/list destekleniyor; yani

```python
L4_FIRST_HLC_RHO_KEY = L4_FIRST_HLC_KEY + "_rho"
MICROCOUNT_SUBKEY    = "STW_m%ip%i_DTW%i" % (STW_MINUS, STW_PLUS, DTW)
```

gibi türetilmiş sabitler de doğru çözülüyor.

**Neden böyle:** `oscNext_L4_variables.py` `from icecube import ...` ile
başlıyor. Notebook onu import etseydi notebook da IceTray'e bağımlı olurdu.
AST ile okuyunca kod çalıştırılmadan sadece isimler alınıyor → notebook
IceTray'siz makinede de açılıyor, **ama** script'te bir anahtarı
değiştirdiğinde notebook otomatik takip ediyor. İki yerin sessizce
birbirinden kopması (script `L4_micro_count` yazıyor, notebook
`L4_microcount` arıyor → her şey NaN) bu şekilde engelleniyor.

**Çıktı yolları:** `OUTPUT_ROOT` = `$OSCNEXT_OUT_ROOT` ya da
`<repo>/L4_output/`. Altında `hdf5/`, `i3/`, `training/`, `models/`.
Hücre `statvfs` ile boş disk de yazdırıyor — 20 GB altındaysa uyarıyor.
Home'da kotan varsa:

```bash
export OSCNEXT_OUT_ROOT=/data/user/$USER/oscNext_L4
```

**Sabitler:** `RNG_SEED=12345` (train/test bölmesi ve permutation importance
tekrarlanabilir olsun diye), `NOISE_CUT=0.70`, `MUON_CUT=0.65` — v00.07 L4
kesim değerleri.

---

## Bölüm 1 — İşleme: `process_L4.py` job'ları

**Ne yapıyor:** `SAMPLES` sözlüğünden (pass3 gerçek yolları) dosya listesi
çıkarıp `process_L4.py` komut satırlarını üretiyor.

```python
SAMPLES = {
  "nue":     {"kind": "genie",   "set": 23800, ...},
  "numu":    {"kind": "genie",   "set": 23799, ...},
  "corsika": {"kind": "corsika", "set": 23694, ...},
  "noise":   {"kind": "noise",   "set": 23813, ...},
}
```

`kind` alanı **hangi MC anahtarlarının book edileceğini** belirliyor:

| kind | ek bayrak | özel durum |
|---|---|---|
| `genie` | `--mc --genie` | `I3GenieInfo.n_flux_events` P frame'e taşınır |
| `corsika` | `--mc --corsika` | `CorsikaWeightMap`, `PolyplopiaPrimary` |
| `noise` | `--noise` | **`MCInIcePrimary` YOK** — ayrı bayrak sebebi bu |
| `data` | (yok) | GCD run numarasından türetilir |

`build_jobs()` → `files_per_job` kadar dosyayı bir job'a topluyor, `L4_jobs.txt`
üretiyor (cluster'a submit edilecek).

`run_jobs()` job'ları yerel process havuzunda paralel çalıştırıyor.
`skip_existing=True` çıktı HDF5 varsa atlıyor → yarım kalan üretimi baştan
başlatmadan tamamlayabilirsin. Her job ayrı subprocess, biri patlarsa diğerleri
devam eder ve sonunda hangilerinin `rc != 0` verdiği raporlanır.

`run_smoke_test()` — **tam üretimden önce mutlaka bu.** Tek dosyada 200 frame
işler. `--apply-cut` KULLANMAZ (sınıflandırıcılar henüz eğitilmedi; kesim
uygularsan eğiteceğin olayları atmış olursun).

> **Dikkat:** bu bölüm IceTray gerektiren tek bölüm. Notebook kernel'i IceTray
> ortamında değilse `run_jobs()` "icecube import edilmiyor" verir. Çözüm:
> `./setup_env.sh kernel` (bkz. `CLAUDE.md`), ya da job'ları terminalden çalıştır.

---

## Bölüm 2 — Booking doğrulaması

**Ne yapıyor:** üretilen HDF5'te beklenen tabloların hepsi var mı, satır
sayıları tutarlı mı?

`EXPECTED_TABLES` her tabloyu hangi örneklerde beklediğimizle eşliyor.
`verify_booking()` dosyayı açıp tablo tablo `OK / MISSING / EMPTY / FEW_ROWS`
veriyor. `verify_all()` her örnekten bir dosya örnekleyip pivot tablo
gösteriyor.

**Neden kritik:** bir tray modülü sessizce başarısız olursa tablo hiç yazılmaz.
Bunu 3 bölüm sonra "bu değişken neden hep NaN" diye keşfetmek yerine burada
yakalıyorsun. Ayrıca `--sub-event-stream` yanlışsa **hiçbir frame işlenmez ve
sessizce boş HDF5 gelir** — bu kontrol onu da yakalar.

Bu adım geçmeden ileri gitme. `L4_iLineFitParams` yoksa noise BDT'nin
`iLineFit_speed` girdisi yok demektir ve model o değişken olmadan eğitilir.

---

## Bölüm 3 — Feature registry

**Ne yapıyor:** her değişkeni `(tablo, kolon)` çifti olarak tanımlıyor.

```python
@dataclass
class Feature:
    name: str            # analiz içindeki isim
    table: str           # HDF5 tablosu
    column: str          # kolon
    alts: Sequence = ()  # isim varyasyonları
    valid_range: tuple   # sağlık kontrolü için
    note: str
```

Frame objesi tipi → HDF5 kolonu eşlemesi:

| Frame objesi | HDF5'te |
|---|---|
| `I3Double` (VICH, accumulated_time, rho) | tek kolon: `value` |
| `I3MapStringDouble/Int` (L3 vars, micro_count) | her map anahtarı ayrı kolon |
| `I3LineFitParams` | `lf_vel`, `lf_vel_x/y/z`, `n_hits` |
| `I3HitStatisticsValues` | `cog_x/y/z`, `z_min`, `z_max`, `z_sigma`, `z_travel`, … |
| `I3FillRatioInfo` | `fill_ratio_from_mean`, `fill_radius_from_mean`, … |
| `I3Particle` (first_hlc) | `x`, `y`, `z`, `time`, `zenith`, … |

**`alts` niye var:** hdfwriter'ın ürettiği kolon isimleri meta-proje sürümüne
göre değişiyor. Örnek: `I3LineFitParams` alanı pass3'te `lf_vel`, eski
varsayım `LFVel` idi (kod içinde "DOGRULANDI" notu var). `resolve_features()`
dosyaya bakıp hangisinin **gerçekten var olduğunu** buluyor. Böylece meta-proje
değişse de notebook kırılmıyor.

Üç grup: `NOISE_FEATURES` (Tablo 11), `MUON_FEATURES` (Tablo 12),
`CANDIDATE_FEATURES` (aday — bölüm 10'da taranacak).

---

## Bölüm 4 — Yükleme

**Ne yapıyor:** HDF5 tablolarını `Run/Event/SubEvent` üzerinden **merge**
ediyor.

```python
merged = merged.merge(df, on=["Run","Event","SubEvent"], how="outer")
```

**Bu bölümün tek önemli fikri bu.** Satır sırasına asla güvenilmiyor. Bir
frame objesi bazı olaylarda yoksa (örn. `HitStatistics` L2'de ~%0.1 olayda
başarısız) satır sırasına dayanan bir okuma **kayar** ve olay A'nın `cog_z`'si
olay B'nin `NchCleaned`'i ile eşleşir. Sessizce yanlış model eğitirsin. Bu tip
analizin en sinsi hatası; merge onu imkansız kılıyor.

`_read_table()` ayrıca `exists` kolonu varsa ona göre filtreliyor (hdfwriter
olmayan frame'ler için `exists=0` satırı yazar).

**`EXTRA_*` sözlükleri** — feature değil ama gerekli olan alanlar:

- `EXTRA_HEADER`: `I3EventHeader` zaman alanları → livetime
- `EXTRA_MC` / `EXTRA_TRUTH`: `OneWeight`, `NEvents`, `n_flux_events` ve
  gerçek enerji/zenith. **Not:** pass3 `I3MCWeightDict`'inde hazır `weight`
  alanı YOK (onu oscNext işlemesi ekliyor, o proje bizde yok) → ham alanlardan
  kendimiz hesaplıyoruz. Ayrıca pass3'te `MCInIcePrimary` yok, truth
  `I3MCWeightDict.PrimaryNeutrino*` içinden alınıyor.
- `EXTRA_CORSIKA`: `simweights`'in ihtiyaç duyduğu tüm `CorsikaWeightMap` alanları
- `EXTRA_NOISE`: `noise_weight.weight` (vuvuzela)

`MAX_FILES` — geliştirirken 20 gibi bir sayı ver, tam üretimde `None`.

---

## Bölüm 5 — Sağlık kontrolü

**Ne yapıyor:** her (örnek × değişken) için beş metrik:

| durum | anlamı |
|---|---|
| `MISSING` | kolon hiç yok → registry ya da booking hatalı |
| `LOW_COVERAGE` | finite oran < %50 → modül olayların yarısında çalışmamış |
| `HAS_INF` | **LightGBM `inf` ile patlar** (NaN'ı doğal işler, inf'i işlemez) |
| `OUT_OF_RANGE` | `valid_range` dışı > %0.1 → tanım/birim hatası |
| `CONSTANT` | tek değer → modül sabit yazmış, bilgi taşımıyor |

`gate()` fonksiyonu noise ve muon girdilerini ayrı ayrı pivot tabloda gösterip
"hepsi OK" ya da ">>> düzeltilmesi gereken var" diyor.

**Bilinen normal kayıp:** `HitStatistics` MuonGun dosyalarının ~%0.1'inde L2'de
başarısız oluyor, nedeni bilinmiyor, o olaylar L3 kesimiyle zaten temizleniyor.
Yani `cog_z`/`z_sigma`/`z_travel` için çok küçük coverage kaybı beklenen.

---

## Bölüm 6 — Yeniden yazılan değişkenlerin doğrulaması

**Bu bölüm bu repodaki en riskli yer.** `oscNext_L4_variables.py` dört şeyi saf
Python'la yeniden yazdı çünkü orijinal projeler (`tau_bdt.I3CutL7Module`,
`analysis.event_selection`, `slc-veto`, `FirstHLC` C++ modülü) modern
IceTray'de yok:

| Değişken | Risk | Belirsizlik |
|---|---|---|
| `VICH_nch` | **Yüksek** | `dt` yönü (`t_COG − t_hit > 0` alındı), veto DOM tanımı |
| `accumulated_time` | Orta | Referans zamanı, fraksiyon eşiği |
| `first_hlc_rho` | Düşük | DOM başına ilk HLC seçimi |
| `separation_in_cogs` | Yüksek ama BDT girdisi değil | Bölme kuralı |

İki doğrulama yolu:

**1. `compare_to_reference(new_h5, ref_h5)`** — altın standart. Elinde v01.03
referans L4 dosyası varsa aynı L3 girdisini iki koddan geçirip olay-olay
karşılaştırıyor: `identical_frac`, `mean_diff`, `p95_absdiff`, `corr` ve
scatter (referans vs yeni, y=x çizgisiyle). `identical_frac ≈ 1.0` istiyorsun.

**2. `sanity_rewritten(feature)`** — referans yoksa fiziksel akıl sağlığı.
νe/νμ vs CORSIKA vs Noise dağılımlarını normalize edip üst üste çiziyor.

**VICH için özel kontrol (hücre 24):** VICH = *Veto Identified Causal Hits* —
COG'a nedensel olarak bağlı, DeepCore veto bölgesindeki hitler. Fizik beklentisi:

- **Nötrino** DeepCore içinde etkileşir → veto bölgesinde önceden hit yok →
  `VICH_nch ≈ 0`, `frac_zero` yüksek
- **Atmosferik muon** yukarıdan geçer → veto DOM'larında COG'dan **önce** hit
  bırakır → `VICH_nch` belirgin kuyruk

Ayrım yoksa iki şüpheli var ve ikisi de kodda:
1. `_vich` içindeki `dt = t_COG - t_hit` yönü ters (veto hit COG'dan ÖNCE
   olmalı, `dt > 0`)
2. Veto DOM tanımı yanlış (`DeepCoreVetoDOMs` listesi)

Bu kontrolü geçmeden model eğitme — VICH muon BDT'sinin en güçlü girdilerinden.

---

## Bölüm 7 — Livetime ve ağırlıklar

Üç ayrı ağırlık kavramı var, karıştırmak kolay:

1. **`w_phys` — fiziksel rate [Hz].** Dağılım çizerken ve kesim performansı
   ölçerken. Data/MC karşılaştırmasının temeli.
2. **`w_train` — eğitim ağırlığı.** Sinyal ve arka planın *toplam* ağırlıkları
   eşitlenir, sonra [0,1]'e çekilir (bölüm 10).
3. **Ağırlıksız sayım.** İstatistiksel yeterlilik için — 50 olaydan oluşan bir
   histogram binini ağırlık düzeltmesi kurtarmaz.

### Livetime

`compute_livetime()`: run başına `max(t) − min(t)`, sonra toplam.
`I3EventHeader.time_start_mjd_day * 86400 + time_start_mjd_sec`.
Detector downtime'ı hesaba katmıyor, ama eğitim için yeterli — ağırlıklar
zaten sonradan sınıf bazında normalize ediliyor.

### GENIE (νe, νμ)

```
w [Hz] = OneWeight × flux(E) / n_flux / n_files
flux(E) = NORM × E^GAMMA        (NORM=2e-2, GAMMA=-3.0)
n_flux  = I3GenieInfo.n_flux_events          (varsa)
        = NEvents × 0.7 (ν) / 0.3 (ν̄)       (yoksa)
```

`n_flux_events` P frame'e `process_L4.py` tarafından `--genie` bayrağıyla
taşınıyor. `[i] ... olayda n_flux_events yok` uyarısı görürsen **`--genie`
bayrağını unutmuşsun** demektir.

`NORM/GAMMA` gerçek atmosferik akı değil, basit bir güç yasası yaklaşımı.
Mutlak oranlar Tablo 13 ile birebir tutmaz ama kendi içinde tutarlı ve
data/MC *şekil* karşılaştırması için yeterli. Gerçek akı için `nuflux`
(Honda) + salınım gerekir.

### Noise (vuvuzela)

```
w = noise_weight.weight × NOISE_NS_SCALE / n_files
NOISE_NS_SCALE = 1e9    # pass3 birimi 1/ns  (pass2 zaten Hz idi → 1.0)
```

Bu 1e9 çarpanı yanlışsa gürültü oranın 9 mertebe kayar. Kontrol: bölüm 7
sonunda gürültü ~36.6 mHz mertebesinde çıkmalı.

### CORSIKA

`simweights` + `GaisserH3a` kullanıyor (`oscnext_rates.py` ile aynı).
`simweights` HDF5 dosyalarını doğrudan okuduğu için dosyalar yeniden açılıyor,
sonuç `Run/Event/SubEvent` üzerinden geri eşleştiriliyor. `simweights` yoksa
`CorsikaWeightMap.Weight / (NEvents × OverSampling)` yaklaşımına düşüyor ve
**uyarıyor** — mutlak oran o durumda güvenilmez.

### Ağırlık sağlık kontrolü

v00.07 pass2'de L3 sonrası beklenen mertebeler (pass3'te birebir tutmaz):

| Bileşen | L3 oranı |
|---|---|
| νe CC | ~0.95 mHz |
| νμ CC | ~3.77 mHz |
| ντ CC | ~0.129 mHz |
| Atm. μ | ~505 mHz |
| Gürültü | ~36.6 mHz |

Mertebe sapıyorsa sırayla şüphelen: (1) `n_files` doğru mu / kısmi job mı,
(2) `NOISE_NS_SCALE`, (3) `n_flux_events` uyarısı çıktı mı, (4) NORM/GAMMA.

Son hücre `log10(w_phys)` histogramı çiziyor ve **`max/toplam`** oranını
yazıyor. **%5'ten büyükse tek bir olay oranını domine ediyor** → istatistik
yetersiz ya da ağırlık hesabı bozuk. Bu tek sayı çok şey söyler.

---

## Bölüm 8 — Türetilmiş değişkenler

Ham değişkenlerin fiziksel motivasyonlu kombinasyonları. Üç grup:

**Muona duyarlı oranlar** — mutlak sayı yerine `NchCleaned`'e normalize:
`veto_over_nch`, `vich_over_nch`, `nabove_over_nch`, `rtveto_over_nch`.
Mantık: 5 veto hit'i 10 DOM'luk olayda çok, 200 DOM'luk olayda az.

**Geometri:** `z_extent = z_max − z_min`, `cog_rho` (string 36'ya
`(46.29, −34.88)` uzaklık — DeepCore merkezi), `cog_z_minus_hlc_z`,
`hlc_to_cog_dist`. Bunlar olayın DeepCore içinde mi başladığını yakalar.

**Gürültüye duyarlı — zaman sıkışması:** `micro_over_nch`,
`micro_l3_over_l4` (L3'ün ve L4'ün farklı pencereli micro count'larının oranı),
`nch_per_ns`, `log_speed`.

> **Temel ilke:** oscNext L3 bilinçli olarak yük bağımlılığından uzaklaştırıldı
> (SPE şablon modelleme hatalarına duyarlılığı azaltmak için yük tabanlı
> değişkenler DOM-sayısı analoglarıyla değiştirildi). Türetilmiş değişkenlerde
> de mümkünse yük yerine DOM sayısı kullan. `qtot_per_dom` bilinçli olarak
> yüke bağlı — data/MC kontrolü şart.

Hepsi **aday**. Bölüm 9 (data/MC) ve 10 (importance) filtresinden geçmeden
nihai listeye alınmaz.

---

## Bölüm 9 — Data/MC uyum kontrolü

**Atlanamaz adım.** Bir değişkende data ile MC uyuşmuyorsa BDT o uyumsuzluğu
öğrenir ve gerçek veride beklediğinden farklı davranır. v00.07'de ~40 değişken
test edildi; nihai listeye girenler **iyi data/MC uyumu *ve* güçlü importance**
gösterenler oldu. Değişken sayısı bilinçli olarak minimize edildi
(overtraining ve model karmaşıklığı).

`plot_data_mc(feature)`: üstte `w_phys` ağırlıklı stacked bileşenler + Total MC
+ Data noktaları (hata çubuğu `sqrt(N_ham) × w_ort`), altta Data/MC oranı,
%±10 bandı yeşil.

`data_mc_score(feature)`: tek sayıya indiriyor — bin başına
`|data/MC − 1|` ortalaması. **Kural: > ~0.2 şüpheli.** Elemeden önce
nedenini anla; bazen uyumsuzluk gerçek bir işleme hatasının belirtisi.

**Muon değişkenleri `after_noise_cut` uygulanmış halde bakılmalı** —
sınıflandırıcı zincirinde muon BDT'si noise kesiminden *sonra* geliyor, o yüzden
onun göreceği örnek bu. Model henüz yoksa `after_noise_cut()` düz kesimlere
düşüyor:

```
n_hit_doms >= 8, STW9000_DTW300Hits >= 2, micro_count >= 2,
fill_ratio >= 0.03, z_sigma >= 8, z_travel >= -50
```

> Bu üretimde **gerçek veri yok** → `MUON_BACKGROUND` CORSIKA'ya düşüyor.
> Data/MC oranı hesaplanamıyor (`data_mc_score` NaN döner). Bu, zincirin en
> zayıf halkası: v00.07 muon BDT'sini gerçek veriyle eğitiyordu.

---

## Bölüm 10 — Korelasyon ve feature importance

**Hyperparametreler Tablo 10'dan** (v00.07 ile aynı):

```python
max_depth=6, num_leaves=25, max_bin=32, min_data_in_leaf=500,
lambda_l1=2.0, lambda_l2=1.0, min_gain_to_split=2.0,
is_unbalance=False, learning_rate=0.05
feature_fraction: noise 0.8 / muon 0.7
train_frac:       noise 1/3 / muon 0.5
```

`max_bin=32` ve `min_data_in_leaf=500` dikkat çekici derecede muhafazakâr —
kasıtlı: az sayıda değişkenle, overtraining'e kapalı, data/MC farklarına az
duyarlı bir model isteniyor.

**`build_training_set()` — ağırlık işleme sırası önemli:**

```python
1. NaN ağırlıklar → 1.0
2. negatifleri kırp  (w >= 0)
3. sınıf içinde %99.9 kuantiline kırp   ← uzun kuyruk tek olayın
                                          gradyanı domine etmesini engeller
4. sınıf bazında normalize: w[label] /= w[label].sum()   ← sinyal ve arka plan
                                                            EŞİT toplam ağırlık
5. w /= w.max()                          ← [0,1], sayısal hata birikimi
```

Adım 4 `is_unbalance=False` ile birlikte anlamlı: dengelemeyi LightGBM'e
bırakmıyoruz, ağırlıklarla kendimiz yapıyoruz.

`X` içinde `inf` → NaN'a çevriliyor (LightGBM NaN'ı öğrenir, inf'te patlar).

**`quick_train()`**: stratified split, 600 round, `early_stopping(50)`,
train ve test AUC'sini yazdırıyor. **`AUC_train − AUC_test > 0.01` → overtraining
uyarısı.** (v00.07'de noise BDT'sinin biraz overtrained olduğu biliniyor;
sebebi gürültü MC istatistiğinin ~1 ay olması.)

**`correlation_plot()`**: Spearman korelasyon matrisi + `|ρ| > 0.85` çiftlerini
listeliyor. Yüksek korelasyonlu çiftten birini atmak modeli sadeleştirir.

**`importance_plot()` — üç metrik:**

| metrik | ne ölçer | biası |
|---|---|---|
| `gain_%` | split'lerin toplam kayıp azaltması | **çok değerli sürekli değişkenler lehine biaslı** |
| `split_%` | kaç kez split'te kullanıldı | aynı yönde biaslı |
| `perm_dAUC` | kolonu karıştırınca AUC ne kadar düşüyor | biassız |

> **Permutation'a daha çok güven.** Gain, `NchCleaned` gibi tamsayı
> değişkenleri haksız yere aşağı çeker çünkü az sayıda olası split noktası var.

`n_repeats=5` ile permutation tekrarlanıp ortalanıyor (tek permutation gürültülü).

**Son iki hücre — "~40 değişken test edildi" adımının karşılığı:**

- *Genişletilmiş tarama:* `MUON_FEATURES + CANDIDATE_FEATURES + DERIVED_NAMES`,
  `data_mc_score > 0.25` olanlar önceden elenmiş halde. Hepsiyle bir model eğit,
  importance sırala.
- *`incremental_scan()`:* importance sırasına göre 1, 2, 3, … n değişkenle
  eğitip test AUC eğrisi çiziyor. **"Kaç değişken yeterli?"** sorusunun cevabı:
  eğrinin düzleştiği yer. Sonrasında eklenen her değişken sadece overtraining
  riski ve data/MC riski getiriyor.

---

## Bölüm 11 — Export

`export()` her sınıflandırıcı için iki dosya yazıyor:

**`L4_<tag>_training.parquet`** — kolonlar: seçilen feature'lar, `label` (1=ν),
`w_train`, `w_phys`, `split` (`train`/`test`), `sample`, `Run/Event/SubEvent`.

`split` kolonu **aynı `RNG_SEED` ve `train_frac` ile** üretiliyor, yani
`train_L4_classifier.py` notebook'takiyle birebir aynı bölmeyi kullanıyor.
Notebook'ta gördüğün AUC ile script'in ürettiği AUC karşılaştırılabilir olur.

**`L4_<tag>_meta.json`** — üretilebilirlik kaydı: feature listesi (**sıra
önemli**, `l4_classifier_module.py` frame'den bu sırayla okuyacak), olay
sayıları, `lgb_params`, kullanılan frame anahtarları (`micro_count` subkey,
VICH hız penceresi, fill ratio yarıçapı), livetime, kaynak glob'ları ve
`code_sha256` — `oscNext_L4_variables.py` + `process_L4.py` dosyalarının
SHA256'sının ilk 16 hanesi.

`code_sha256` şu soruyu cevaplıyor: *"bu modeli hangi kodla ürettim?"*
Altı ay sonra bir modeli yeniden üretmen gerektiğinde tek dayanağın bu.

Son hücre çalıştırılacak komutları yazdırıyor:

```bash
python train_L4_classifier.py --tag noise --input .../L4_noise_training.parquet --outdir .../models
python train_L4_classifier.py --tag muon  --input .../L4_muon_training.parquet  --outdir .../models
python process_L4.py --gcd ... --input ... --output-hdf5 ... --apply-cut --model-dir .../models
```

---

## Kontrol listesi (notebook sonundaki)

**İşleme** — smoke test hatasız, `verify_booking()` hepsi OK,
`--sub-event-stream` doğru, tüm job'lar tamam.

**Değişkenler** — sağlık raporunda BDT girdileri OK, `VICH_nch` muonlarda
belirgin yüksek, referans varsa `compare_to_reference()` çalıştırıldı.

**Ağırlıklar** — livetime makul, MC oranları beklenen mertebede,
`n_files` doğru.

**Eğitim** — data/MC sapması yüksek değişkenler elendi, `AUC_train − AUC_test
< 0.01`, muon arka planı gerçekten noise kesiminden geçti.

---

# pybdt nasıl çalışır

> Bu bölüm bu branch'in **aktif** yolunu anlatıyor. `pybdt/` kaynağı repoya
> vendor edilmiş, `pybdt_train.py` + `pybdt_classifier_module.py` +
> `oscNext_L4_pybdt.ipynb` bu yolu kullanıyor. LightGBM yolu
> (`train_L4_classifier.py`, `l4_classifier_module.py`) referans olarak
> duruyor — teknik notun (v00.074, §3.6.1) resmi yöntemi odur.

## Algoritma: AdaBoost + karar ağaçları

pybdt, LightGBM'in kullandığı **gradient boosting**'den farklı olarak
**AdaBoost** uygular. Fark önemli:

| | pybdt (AdaBoost) | LightGBM (gradient boosting) |
|---|---|---|
| Ağaç neyi öğrenir | yanlış sınıflandırılan **olayların ağırlığı artırılır** | önceki modelin **gradyanı/residüeli** |
| Ağaç yapısı | derinliğe göre simetrik büyütme + budama | leaf-wise, `num_leaves` sınırlı |
| Split kriteri | `separation_type` (Gini vb.) | histogram tabanlı gain |
| Çıktı | ağırlıklı oy toplamı, ~[−1, +1] | log-odds → sigmoid → [0, 1] |
| Eksik değer | doğal desteği **yok** | NaN'ı bir yön olarak öğrenir |
| Aşırı uyum kontrolü | budama (pruner) + KS testi | early stopping + λ regularizasyon |

**AdaBoost döngüsü:**

```
w_i = 1/N   (başlangıç)
her t = 1..num_trees için:
    h_t  = ağırlıklı olaylarla bir karar ağacı eğit
    err  = Σ_{yanlış} w_i  /  Σ w_i
    α_t  = beta · ln((1 − err) / err)
    w_i ← w_i · exp(α_t)   yanlış sınıflandırılan olaylar için
    w yeniden normalize
skor(x) = Σ_t α_t · h_t(x)  /  Σ_t α_t      →  [−1, +1]
```

Kilit sezgi: **her yeni ağaç bir öncekinin hata yaptığı olaylara odaklanır.**
Sinyal ile arka planın karıştığı sınır bölgesi giderek daha yüksek ağırlık
alır. `beta` LightGBM'deki `learning_rate`'in karşılığı — küçükse daha yavaş
ama daha kararlı.

**Tablo 10 parametreleri pybdt'ye doğrudan taşınamaz.** `num_leaves`,
`max_bin`, `lambda_l1/l2`, `min_gain_to_split` LightGBM'in native isimleri;
pybdt'de karşılıkları yok. oscNext için optimize edilmiş bir pybdt parametre
kümesi **mevcut değil** — kendin ayarlaman gerekiyor.

## Bu repodaki gerçek API

`pybdt_train.py:71` — dikkat, learner ağırlık kolonlarını **kurucuda** alıyor
ve `dtlearner` ayrı yaratılmıyor, learner'ın içinden erişiliyor:

```python
from pybdt import ml

# BDTLearner(degiskenler, sinyal_agirlik_kolonu, arkaplan_agirlik_kolonu)
learner = ml.BDTLearner(list(features), weight_col, weight_col)

# zayif ogrenici learner'in ICINDE
dt = learner.dtlearner
dt.max_depth            = 3
dt.min_split            = 500     # yaprakta minimum olay
dt.num_cuts             = 100     # degisken basina denenen kesim sayisi
dt.num_random_variables = 3       # her split'te rastgele degisken alt kumesi
dt.linear_cuts          = True    # False -> nonlinear_histogram

# boost parametreleri
learner.num_trees          = 300
learner.beta               = 0.7
learner.frac_random_events = 0.5  # bagging
learner.use_purity         = True

# budama -- LightGBM'de karsiligi yok, pybdt'de ONEMLI
learner.add_before_pruner(ml.SameLeafPruner())
learner.add_before_pruner(ml.CostComplexityPruner(35))

bdt = learner.train(sig_train, bg_train)
util.save(bdt, "L4_noise.bdt")
```

`pybdt_train.py` bunları CLI bayrağı olarak alıyor ve **yalnızca açıkça
verilenleri** set ediyor; gerisi pybdt'nin kendi varsayılanlarında kalıyor.
`effective_params()` (satır 99) fiilen kullanılan değerleri okuyup `.json`'a
yazıyor — "bu modeli hangi parametrelerle eğittim" sorusunun cevabı orada.

**`use_purity`** — skorların yaprak saflığıyla ağırlıklandırılması. Eğitimde
açıksa **okumada da açık olmalı**: `score_expr()` (satır 152) `use_purity`
ise `pscores`, değilse `scores` döndürüyor. Karıştırılırsa skorlar sessizce
yanlış olur.

## Doğrulama — `Validator`

pybdt'nin en güçlü yanı bu. `build_validator()` (satır 121) dört DataSet'i
(sig/bg × train/test) ekliyor, `make_plots()` üç grafik üretiyor:

- `_overtrain.png` — train ve test skor dağılımları üst üste. **KS testi
  otomatik**: `p_KS < 0.01` ise uyarı basıyor (pybdt dokümantasyonunun eşiği,
  `man_overtraining.rst`). Uyarı çıkarsa `num_trees` düşür ya da
  `prune_strength` artır.
- `_dist.png` — skor dağılımı
- `_rate.png` — kesim değerine karşı rate

LightGBM yolunda bunu elle yapıyorduk (train/test AUC farkı > 0.01 kontrolü);
pybdt hazır getiriyor ve KS testi AUC farkından daha hassas.

## Frame'e uygulama

`pybdt_classifier_module.py` → `PyBDTClassifier` tray modülü. `FEATURE_MAP`'i
`l4_classifier_module.py`'den import ediyor — **mapping tek yerde kalsın
diye**. O modül `lightgbm`'i sadece kendi `load_model()`'ı içinde import
ettiği için bu bağımlılık lightgbm gerektirmiyor.

## İki tuzak

**1. Import yolu.** `import pybdt` — `from icecube import pybdt` **değil**.
pybdt diğer IceTray projelerinin aksine `icecube` isim alanında değil,
bağımsız üst düzey bir paket (kendi kaynağı da böyle import ediyor, bkz.
`pybdt/python/pybdtmodule.py`). Bu proje geçmişinde bir kez "pybdt
derlenmemiş" sanılmasına yol açtı. `icetray_env.require_pybdt()` artık bu
hatayı yakalayıp doğru kullanımı söylüyor.

**2. `.ds` dosyalarında BDT girdisi olmayan kolonlar var** (`w_phys` —
fiziksel ağırlık). Eğitime `--features` **açıkça** geçilmeli; `pybdt_train.py`
verilmezse `RESERVED_COLS`'u dışarıda bırakıyor. Bu koruma olmasa model
fiziksel ağırlığı bir değişken sanıp öğrenirdi — sessiz ve ciddi bir hata.

## Neden LightGBM referans olarak duruyor

- NaN'ı doğal işliyor (IceCube verisinde eksik reco yaygın; pybdt'de yok)
- `.txt` native formatı sklearn/joblib gerektirmiyor
- Teknik notun resmi yöntemi bu — sonuçları resmi analizle karşılaştırmak
  gerekirse referans lazım

pybdt'nin karşılığında getirdiği: IceCube'un kendi aracı olması, hazır
`Validator` altyapısı ve KS tabanlı overtraining kontrolü.
