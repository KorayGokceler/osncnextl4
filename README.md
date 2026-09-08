# oscNext L4 — değişken üretimi ve BDT eğitimi (pass3)

L3 `.i3` dosyalarından L4 değişkenlerini üretir, HDF5'e book eder ve
noise + muon sınıflandırıcılarını eğitir.

**İki ayrı BDT yolu var** — ayrıntılı gerekçe için `CLAUDE.md`:

| Yol | Motor | Durum |
|---|---|---|
| **pybdt** | IceCube'un kendi AdaBoost kütüphanesi | Aktif geliştirilen yol. Kaynaktan derlendi. |
| LightGBM | gradient boosting | Referans olarak duruyor. Teknik notun (v00.074) resmi yöntemi budur. |

## Dosyalar

| Dosya | Nerede çalışır | Ne yapar |
|---|---|---|
| `oscNext_L4_variables.py` | IceTray | L4 değişkenlerini hesaplayan tray segmentleri |
| `process_L4.py` | IceTray | `.i3` → L4 değişkenleri → `.hdf5` |
| `simple_booker.py` | IceTray | `hdfwriter` yoksa fallback booker |
| `oscNext_L4_pybdt.ipynb` | pybdt build'i | **ANA ARAYÜZ** — uçtan uca tüm süreç |
| `pybdt_train.py` | pybdt build'i | `.ds` → BDT eğitimi + doğrulama grafikleri |
| `pybdt_classifier_module.py` | pybdt build'i | Eğitilmiş pybdt modelini frame'e uygular |
| `train_L4_classifier.py` | herhangi | *(referans)* Parquet → LightGBM modeli |
| `l4_classifier_module.py` | IceTray | *(referans)* LightGBM modelini frame'e uygular + `FEATURE_MAP` |
| `oscNext_L4_feature_engineering.ipynb` | IceTray | *(referans)* Eski uçtan uca arayüz |
| `diagnose_env.py` | IceTray | Ortamda ne var ne yok (pybdt kontrolü dahil) |
| `reference/` | — | pass2 teknik notu (PDF) + gerçek pass3 L3 scripti |
| `pybdt/` | — | pybdt kaynağı (okuma/referans; derleme buradan YAPILMAZ) |

## Kurulum

### 1. Repoyu klonla
```bash
git clone https://github.com/KorayGokceler/osncnextl4.git l4
cd l4
git checkout claude/oscnext-l4-scripts-35min0
```

### 2. pybdt'yi derle (bir kez)

`pybdt`, cvmfs'teki hazır `py3-v4.4.2` dağıtımında **derlenmiş gelmiyor**
(`BUILD_PYBDT` bayrağı kapalı). Kendi build'ini yapman gerekiyor.
`/cvmfs` salt-okunur, ona hiç dokunulmuyor — her şey kendi alanına yazılır.

```bash
mkdir -p /data/user/$(whoami)/icetray_build
cd /data/user/$(whoami)/icetray_build
```

`icetray` kaynağını (v1.17.0) buraya `src/` olarak koy — GitHub'dan ZIP
indirip açmak en kolayı (`icecube/icetray` private repo, git klonlamak
SSO yetkilendirmesi isteyebilir).

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
mkdir build && cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_PYBDT=ON
make -j8 pybdt
```

Yalnızca `serialization`, `icetray`, `dataclasses` ve `pybdt` hedefleri
derlenir (tüm meta-proje değil), birkaç dakika sürer. Cobalt paylaşılan
bir makine olduğu için `-j64` yerine `-j8` gibi ölçülü bir değer kullan.

### 3. Ortamı aç (her oturumda)

**İki adım da gerekli, bu sırayla:**
```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
cd /data/user/$(whoami)/icetray_build/build
./env-shell.sh
```

Doğrula:
```bash
python -c "import pybdt; print('OK', pybdt.__file__)"
```

> **Dikkat:** `import pybdt` — `from icecube import pybdt` **değil**.
> pybdt, diğer IceTray projelerinin aksine `icecube` isim alanına dahil
> değildir.

### 4. Jupyter (isteğe bağlı)

Jupyter'i **bu ortamın içinden** başlat; içinde açtığın her terminal ve
notebook kernel'i ortamı miras alır, tekrar `env-shell` gerekmez:
```bash
cd ~/l4
python -m jupyter lab --no-browser --port=8896
```
Yerelden: `ssh -L 8896:localhost:8896 <kullanıcı>@cobalt.icecube.wisc.edu`

Jupyter sunucusu yeniden başlarsa 3. adımı tekrarlaman gerekir.

## Kullanım

### L3 → L4 işleme (HDF5 üretimi)
```bash
python process_L4.py \
    --gcd  /data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz \
    --input "/data/ana/LE/oscNext/pass3/genie/level3/23800/*.i3.zst" \
    --output-hdf5 L4_output/hdf5/nue/L4_nue.hdf5 \
    --mc --genie
```
Örnek türüne göre bayrak: `--genie`, `--corsika`, `--noise`, `--muongun`.
`--apply-cut` **kullanma** — modeller eğitilmeden önce tüm olaylar book
edilmeli.

Üretilen HDF5'in gerçek sütun isimlerini görmek için notebook'un
2. bölümünü çalıştır (`dump_tables`) — tüm tabloları ve kolonları listeler.

### Uçtan uca: notebook

Normalde her şeyi `oscNext_L4_pybdt.ipynb` üzerinden yaparsın — işleme,
ağırlıklar, `.ds` üretimi, eğitim, doğrulama, kesim seçimi hepsi orada,
sırayla çalıştırılacak bölümler halinde. Aşağıdaki komutlar notebook'un
perde arkasında çağırdığı adımlar.

### BDT eğitimi (pybdt)
```bash
python pybdt_train.py --name L4_noise --outdir models_pybdt \
    --sig-train ds/noise_sig_train.ds --bg-train ds/noise_bg_train.ds \
    --sig-test  ds/noise_sig_test.ds  --bg-test  ds/noise_bg_test.ds \
    --num-trees 300 --depth 3 --beta 0.7 --prune-strength 35 \
    --frac-random-events 0.5 --use-purity
```

Üretir:
- `L4_noise.bdt` — eğitilmiş model
- `L4_noise.validator` — skorları önceden hesaplanmış `Validator`
- `L4_noise.json` — değişkenler, **fiilen kullanılan** hiperparametreler, KS p değerleri
- `L4_noise_overtrain.png` / `_dist.png` / `_rate.png`

Overtraining kontrolü otomatik: pybdt'nin KS testi çalışır, `p_KS < 0.01`
ise uyarı basar (pybdt dokümantasyonunun eşiği).

`.ds` dosyaları notebook'un 6. bölümünde üretilir (HDF5 → numpy →
`pybdt.ml.DataSet` → `util.save`).

> **Dikkat:** `.ds` içinde BDT girdisi olmayan kolonlar da var (`w_phys`).
> Eğitime `--features` **açıkça** geçilmeli — yoksa model fiziksel
> ağırlığı bir değişken sanıp öğrenebilir.

### Modeli frame'e uygulama
```python
from pybdt_classifier_module import PyBDTClassifier

tray.Add(PyBDTClassifier, "noise_clf",
         ModelFile="models_pybdt/L4_noise.bdt",
         OutputKey="L4_NoiseClassifier_pybdt")
```

## Önemli notlar

**Veri repoya girmez.** `.gitignore` `L4_output/` ve tüm veri
uzantılarını engelliyor. Home dizininde kotan varsa çıktıları başka yere
yönlendir:
```bash
export OSCNEXT_OUT_ROOT=/data/user/<kullanıcı>/oscNext_L4
```

**Notebook çıktılarını temizle** (commit'lemeden önce):
```bash
pip install --user nbstripout && nbstripout --install
```

## Girdi verisi (pass3)

| Örnek | Set | Yol |
|---|---|---|
| νe | 23800 | `/data/ana/LE/oscNext/pass3/genie/level3/23800/` |
| νμ | 23799 | `/data/ana/LE/oscNext/pass3/genie/level3/23799/` |
| CORSIKA | 23694 | `/data/ana/LE/oscNext/pass3/corsika/level3/23694/` |
| Noise | 23813 | `/data/ana/LE/oscNext/pass3/noise/level3/23813/` |

ντ ve gerçek dedektör verisi bu üretimde yok:
- Sinyal = νe + νμ (ντ CC toplam sinyalin ~%3'ü)
- Muon BDT arka planı gerçek veri yerine CORSIKA

## Durum

- [x] Ortam doğrulandı (`hdfwriter` yok → SimpleBooker; `oscNext` projesi yok)
- [x] pybdt kaynaktan derlendi ve çalışıyor
- [x] νe işleme çalışıyor, tüm tablolar book ediliyor
- [x] Uçtan uca notebook yazıldı (`oscNext_L4_pybdt.ipynb`)
- [ ] Notebook hiç çalıştırılmadı → HDF5 sütun isimleri doğrulanmadı,
      3. bölümdeki `REGISTRY` düzeltme gerektirebilir
- [ ] νμ / CORSIKA / noise işleme denenmedi
- [ ] Sütun isimleri kesinleştirilmedi
- [ ] Yeniden yazılan değişkenler (VICH, accumulated_time) doğrulanmadı
- [ ] Ağırlıklar doğrulanmadı
- [ ] Model eğitilmedi
