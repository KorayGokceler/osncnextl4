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
`train_L4_classifier.py` → `L4_{tag}_model.txt`/`.json` →
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
- [ ] Sütun isimleri kesinleştirilmedi
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
   alıyoruz. Ayrıca not DC Filter'ın kendi SRT temizlemesini kullanıyor,
   biz ham `SplitInIcePulses`.
2. **accumulated_time** — **doğrulandı**: Tablo 12 "Time to reach 75% of an
   event's charge in the cleaned pulse series" diyor; kod `fraction=0.75` ve
   `cleaned_pulses` kullanıyor. Fraksiyon ve seri artık tahmin değil.
   **Kalan açık:** referans zamanı — kod `t[idx] − t[0]` (ilk pulse) alıyor,
   not sıfır noktasını söylemiyor (tetikleme zamanı da olabilirdi).
   `separation_in_cogs` BDT girdisi değil, düşük öncelik.
3. **FullTimeLengthRatio yönü** — not (Tablo 11) oranın yönünü söylemiyor.
   Ama temizlenmiş seri temizlenmemişin alt kümesi olduğu için
   `cleaned/uncleaned ∈ [0,1]` sınırlı ve fiziksel; kodun aldığı yön bu.
   Ayrıca not bunu **L3 değişkeni** olarak listeliyor
   (`IC2018_LE_L3_Vars.FullTimeLengthRatio`); pass3 L3 map'inde oran yok,
   bileşenleri var (`CleanedFullTimeLength`, `UncleanedFullTimeLength`).
3a. **Muon BDT'de eksik girdi (düzeltildi).** Tablo 12 **10** değişken
   listeliyor, `MUON_FEATURES`'ta **9** vardı — `NchCleaned` atlanmıştı
   (noise listesinde olduğu için gözden kaçmış). Eklendi. Benzersiz BDT
   değişkeni: 14 (5 noise + 10 muon, `NchCleaned` ortak).
3c. **`noise_weight` kolonu (düzeltildi).** `AUX`'ta `("noise_weight",
   "value")` yazıyordu; eski LightGBM notebook'unda doğrulanmış hali
   `("noise_weight", "weight")`. Yanlış olduğu için gürültü ağırlığı
   **tamamen NaN** kalıyordu → noise örneğinin `w_phys`'i sıfır olurdu.
   Düzeltildi, ikisi de `ALTS`'te.
3b. **`iLineFit_speed` kolonu — kontrol edilmeli.** Tablo 11 girdiyi
   `L4_iLineFit.speed` (I3Particle alanı) diye veriyor; pybdt notebook'u
   `L4_iLineFitParams.LFVel` okuyor, LightGBM notebook'u ise pass3 kolonunun
   `lf_vel` olduğunu "DOĞRULANDI" diye not düşmüş. Yanlışsa **sessiz NaN**.
   Bölüm 2'de `dump_tables(..., only=["iLineFit"])` ile 10 saniyede çözülür.
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
  `train_L4_classifier.py`'nin aksine pybdt `icecube.*` namespace
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
  sklearn / lightgbm / pandas kullanmaz, `train_L4_classifier.py`'den
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
