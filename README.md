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
| `icetray_env.py` | her yer | **IceTray/pybdt köprüsü** — import başarısız olursa sebebini söyler |
| `setup_env.sh` | shell | ortamı bul / shell aç / tek komut çalıştır / Jupyter kernel |
| `scan_files.py` | IceTray | bozuk `.i3.zst` dosyalarını bul, sağlam liste üret |
| `oscNext_L4_variables.py` | IceTray | L4 değişkenlerini hesaplayan tray segmentleri |
| `process_L4.py` | IceTray | `.i3` → L4 değişkenleri → `.hdf5` |
| `simple_booker.py` | IceTray | `hdfwriter` yoksa fallback booker |
| `oscNext_L4_pybdt.ipynb` | pybdt build'i | **ANA ARAYÜZ** — uçtan uca tüm süreç |
| `pybdt_train.py` | pybdt build'i | `.ds` → BDT eğitimi + doğrulama grafikleri |
| `pybdt_classifier_module.py` | pybdt build'i | Eğitilmiş pybdt modelini frame'e uygular |
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

...ya da repodaki sarmalayıcı (build'i kendisi bulur):
```bash
cd ~/l4 && ./setup_env.sh shell
```

Doğrula:
```bash
python -c "import pybdt; print('OK', pybdt.__file__)"
# ya da hepsini birden (icecube + I3Tray + pybdt):
./setup_env.sh run python icetray_env.py
```

> **Dikkat 1:** `import pybdt` — `from icecube import pybdt` **değil**.
> pybdt, diğer IceTray projelerinin aksine `icecube` isim alanına dahil
> değildir. `icetray_env.require_pybdt()` bu hatayı yakalayıp söylüyor.

> **Dikkat 2:** `env-shell.sh` **yeni bir shell açar**. Bir script'in içinde
> ard arda `./env-shell.sh` ve `python ...` yazarsan ikinci satır o
> shell'den çıkıldıktan sonra, yani ortam olmadan çalışır. Tek komut için:
> ```bash
> ./env-shell.sh -- python process_L4.py ...     # ya da
> ./setup_env.sh run python process_L4.py ...
> ```

`setup_env.sh` build'i şu sırayla arar: `$OSCNEXT_I3_BUILD` → `$I3_BUILD` →
`/data/user/$USER/icetray_build/build` → `/data/user/$USER/*/build` →
`~/*/build` → cvmfs metaproject'leri. Başka yerdeyse:
```bash
export OSCNEXT_I3_BUILD=/tam/yol/build
```

### 4. Jupyter (isteğe bağlı)

Jupyter'i **bu ortamın içinden** başlat; içinde açtığın her terminal ve
notebook kernel'i ortamı miras alır, tekrar `env-shell` gerekmez:
```bash
cd ~/l4
python -m jupyter lab --no-browser --port=8896
```

Jupyter'i ortam dışından başlattıysan notebook `icecube`/`pybdt` göremez.
O durumda kernel'i bir kez kaydet ve notebook'ta seç
(**Kernel > Change Kernel > "IceTray (oscNext L4)"**):
```bash
./setup_env.sh kernel
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

**Smoke test** (`--n` ile küçük deneme):
```bash
python process_L4.py --gcd ... --scan off \
    --input /data/.../genie_NuE_IC86.023800.000000.i3.zst \
    --output-hdf5 L4_output/hdf5/nue/L4_nue_smoke.hdf5 --mc --genie --n 200
```

> **`--n` FRAME sayar, olay değil.** Akışta G/C/D, Q ve P frame'leri var;
> P frame'lerin de ancak bir kısmı `--sub-event-stream`'e uyup L3 kesimini
> geçiyor. `--n 200` ile ~60 olay book edilmesi normal. Çıktı kademeyi
> gösteriyor:
> ```
> Physics frame           : 98
>   InIceSplit            : 98  (100.0%)
>   L3 kesimi sonrasi     : 60  (61.2%)
> ```
> `InIceSplit` satırı 0 ise `--sub-event-stream` yanlış; L3 satırı 0 ise
> girdi L3 çıktısı değil (test için `--no-l3-cut`).
>
> `--n` verildiğinde tray erken durur, dosya listesinin tamamı okunmaz —
> smoke test'te tek dosya verin ve `--scan off` kullanın.

Üretilen HDF5'in gerçek sütun isimlerini görmek için notebook'un
2. bölümünü çalıştır (`dump_tables`) — tüm tabloları ve kolonları listeler.

### Bozuk girdi dosyaları

pass3 üretiminde yarım yazılmış `.i3.zst` dosyaları var. `I3Reader` böyle bir
dosyaya gelince

```
FATAL (I3Reader): Error reading .../genie_NuMu_IC86.023799.000046.i3.zst
                  at frame 4: input stream error!
```

atıp **tüm tray'i öldürüyor** — 100 dosyalık bir job'da tek bozuk dosya
yüzünden 99 sağlam dosyanın işlenmesi boşa gidiyor ve geride açılamayan
yarım bir HDF5 kalıyor. Ardından gelen
`Table 'Data_quality_bool' is still connected ... This is a BUG!`
mesajı **bunun sonucu**, ayrı bir hata değil.

`process_L4.py` iki katmanlı koruma yapıyor:

1. **Ön tarama** (`--scan quick`, varsayılan açık) — her dosyanın ilk 25
   frame'i okunur, açılmayanlar elenir. Kesik dosyalar tipik olarak ilk
   frame'lerde patladığı için saniyeler içinde yakalanır.
2. **Çalışma anı** (`--retries 3`, varsayılan) — tray yine de patlarsa hata
   mesajından dosya adı çıkarılır, kara listeye yazılır, yarım HDF5 silinir
   ve o dosya hariç yeniden denenir.

```
--scan quick   # varsayılan: ilk N frame (--scan-frames, varsayılan 25)
--scan full    # her frame okunur -- dosyanın ortasında bozulma varsa gerekli
--scan off     # tarama yok (liste zaten temizse)
--retries 0    # çalışma anı yeniden denemesi kapalı
```

Elenen dosyalar `<çıktı>.hdf5.badfiles.txt` içine yazılır.

**Üretimden önce bir kez tara (önerilen).** Her job aynı taramayı tekrar
etmesin diye set başına bir kez tarayıp sağlam listeyi kaydet:

```bash
python scan_files.py --good-list good_23799.txt \
    '/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst'

python process_L4.py --input-list good_23799.txt --scan off \
    --gcd ... --output-hdf5 ... --mc --genie
```

`scan_files.py` bozuk dosya bulursa çıkış kodu 1 döner — shell script'ten
kontrol edilebilir.

> Patlayan job'lardan kalan yarım HDF5'ler açılamaz, silin:
> `rm -f L4_output/hdf5/numu/L4_numu.hdf5`

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

- [x] Ortam doğrulandı (`oscNext` projesi yok; `slc-veto` yok)
- [x] pybdt kaynaktan derlendi ve çalışıyor
- [x] **Kendi build'de `icecube.hdfwriter` VAR** — `Booking: icecube.hdfwriter`.
      SimpleBooker fallback'i devrede değil (cvmfs metaproject'inde yoktu)
- [x] IceTray/pybdt import katmanı (`icetray_env.py` + `setup_env.sh`)
- [x] Bozuk girdi dosyalarına dayanıklılık (`--scan` + `--retries`)
- [x] νe işleme çalıştı — 100 dosya → 256 799 olay, 149 MB, 1019 s
- [x] CORSIKA işleme çalıştı — 500 dosya → 6 462 olay, 4.2 MB, 2578 s
- [x] Uçtan uca notebook yazıldı (`oscNext_L4_pybdt.ipynb`)
- [x] Sütun isimleri kesinleştirildi (14/14 BDT girdisi bulundu)
- [x] L3 **girdisi** gerçek dosyayla doğrulandı — `IC2018_LE_L3_Vars` var,
      14 kolonun tam listesi `CLAUDE.md`'de. Referans L3 scripti bu map'i
      hiç anmıyor, yani üretildiği bir çıkarımdı; artık değil.
- [ ] νμ / noise yeniden çalıştırılmalı — bozuk dosya yüzünden yarım kaldı
      (`--scan` düzeltmesinden sonra)
- [ ] CORSIKA istatistiği az görünüyor (dosya başına ~13 olay); muon BDT
      arka planı bu, eğitim öncesi ağırlık kontrolü şart
- [x] VICH / accumulated_time teknik notla karşılaştırıldı
      (`TEKNIK_NOT_KARSILASTIRMA.md`); VICH'te bulunan COG sapması düzeltildi
- [ ] Yeniden yazılanlar hâlâ **referans çıktıyla** (orijinal C++) karşılaştırılmadı
- [ ] Ağırlıklar doğrulanmadı
- [ ] Model eğitilmedi
