# oscNext L4 — değişken üretimi ve BDT eğitimi (pass3)

L3 `.i3` dosyalarından L4 değişkenlerini üretir, HDF5'e book eder, feature
engineering yapar ve noise + muon sınıflandırıcılarını eğitir.

## Dosyalar

| Dosya | Nerede çalışır | Ne yapar |
|---|---|---|
| `oscNext_L4_feature_engineering.ipynb` | IceTray | **Ana arayüz** — tüm süreç buradan |
| `oscNext_L4_variables.py` | IceTray | L4 değişkenlerini hesaplayan tray segmentleri |
| `process_L4.py` | IceTray | `.i3` → L4 değişkenleri → `.hdf5` |
| `train_L4_classifier.py` | herhangi | Parquet → LightGBM modeli (`.txt` + `.json`) |
| `l4_classifier_module.py` | IceTray | Modeli frame'e uygular (`I3Classifier` yerine) |
| `simple_booker.py` | IceTray | `hdfwriter` yoksa fallback booker |
| `diagnose_env.py` | IceTray | Ortamda ne var ne yok |
| `dump_columns.py` | herhangi | Üretilen HDF5'in tam sütun isimleri |
| `inspect_classifier_api.py` | IceTray | Referans model dosyasının şemasını çözer |

## Kurulum (cobalt)

```bash
# 1. SSH anahtarı (bir kez)
ssh-keygen -t ed25519 -C "cobalt"
cat ~/.ssh/id_ed25519.pub
#    -> GitHub > Settings > SSH and GPG keys > New SSH key

# 2. Klonla
cd ~
git clone git@github.com:<kullanici>/<repo>.git l4
cd l4

# 3. IceTray ortamı
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/metaprojects/icetray/v1.17.0/env-shell.sh

# 4. Ortamı doğrula
python diagnose_env.py
```

Sunucular arası SSH agent forwarding kullanıyorsan anahtar üretmene gerek yok:
`ssh -A cobalt` ile bağlan, yerel anahtarın kullanılır.

## Çalıştırma

```bash
jupyter lab --no-browser --port 8888
# yerelden:  ssh -L 8888:localhost:8888 <kullanici>@cobalt.icecube.wisc.edu
```

Notebook'u sırayla çalıştır. Bölüm sırası ve ne yaptıkları notebook içinde.

## Önemli

**Veri repoya girmez.** `.gitignore` `L4_output/` ve tüm veri uzantılarını
engelliyor. Çıktılar `L4_output/` altında kalır, sadece kod versiyonlanır.

**Notebook çıktılarını temizle.** Grafikler notebook'u MB'lara şişirir ve her
commit'te diff çıkarır:

```bash
pip install --user nbstripout
nbstripout --install        # repo içinde bir kez; commit'te otomatik temizler
```

**Disk.** `L4_output/hdf5/` onlarca GB olabilir. Home dizininde kotan varsa:

```bash
export OSCNEXT_OUT_ROOT=/data/user/<kullanici>/oscNext_L4
```

Notebook bu değişkeni okur; ayarlanmışsa çıktılar oraya gider, repo temiz kalır.

## Girdi verisi (pass3)

| Örnek | Set | Yol |
|---|---|---|
| νe | 23800 | `/data/ana/LE/oscNext/pass3/genie/level3/23800/` |
| νμ | 23799 | `/data/ana/LE/oscNext/pass3/genie/level3/23799/` |
| CORSIKA | 23694 | `/data/ana/LE/oscNext/pass3/corsika/level3/23694/` |
| Noise | 23813 | `/data/ana/LE/oscNext/pass3/noise/level3/23813/` |

ντ ve detektör verisi bu üretimde yok. Sonuçları:
- Sinyal = νe + νμ (ντ CC toplam sinyalin ~%3'ü)
- Muon BDT arka planı gerçek veri yerine CORSIKA

## Durum

- [x] Ortam doğrulandı (`hdfwriter` var, `oscNext` projesi yok, `slc-veto` yok)
- [x] νe işleme çalışıyor, tüm tablolar book ediliyor
- [ ] νμ / CORSIKA / noise işleme denenmedi
- [ ] Sütun isimleri kesinleştirilmedi (`dump_columns.py`)
- [ ] Yeniden yazılan değişkenler doğrulanmadı (VICH, accumulated_time)
- [ ] Ağırlıklar doğrulanmadı
- [ ] Model eğitilmedi

## Notlar

`icecube.oscNext` bu meta-projede yok, bu yüzden şunlar yeniden yazıldı:
`I3Classifier` → `l4_classifier_module.py`, `oscNext_cut` → `L3_oscNext_bool`
okuma, `calc_rho_36` → gömülü string 36 koordinatları.

Eski projeler (`tau_bdt`, `analysis.event_selection`) de yok:
`_vich`, `_accumulated_time`, `_separation_in_cogs` saf Python'la yazıldı ve
**henüz referansla doğrulanmadı** — `oscNext_L4_variables.py` içindeki
uyarılara bakın.
