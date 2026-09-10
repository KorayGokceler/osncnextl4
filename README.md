# oscNext L4 — Level 3 → Level 4

IceCube oscNext (düşük enerji nötrino) analizinin **L3 → L4** adımı.
L3 `.i3` dosyalarından L4 ayırt edici değişkenlerini hesaplar, HDF5'e book
eder ve iki LightGBM sınıflandırıcısı eğitir:

- **noise** — saf gürültü (vuvuzela) reddi
- **muon** — atmosferik muon reddi (arka plan: CORSIKA)

Referans: oscNext technical note v00.07 (bölüm 3.4–3.6, Tablo 10–13).
Yöntem notunkiyle aynı: LightGBM, Tablo 10 hiperparametreleri.

## Kurulum

**1. IceTray ortamı.** `icecube.*` import'ları bir IceTray build'inin
`env-shell.sh`'i içinden çalışır. Arama sırası: `$OSCNEXT_I3_BUILD` →
`$I3_BUILD` → `/data/user/$USER/icetray_build/build` →
`/data/user/$USER/*/build` → `~/*/build` → cvmfs metaproject'leri.

```bash
./setup_env.sh find          # ne bulunuyor
./setup_env.sh shell         # ortam icinde shell ac
./setup_env.sh run python diagnose_env.py    # tek komut
```

> **Tuzak:** `env-shell.sh` **yeni bir shell açar**. Script içinde ard arda
> `./env-shell.sh` ve `python ...` yazarsan ikinci satır ortam olmadan
> çalışır. Tek komut için `./env-shell.sh -- python ...` ya da
> `./setup_env.sh run python ...`.

**2. lightgbm.** Eğitimin ve uygulamanın tek ML bağımlılığı.

```bash
./setup_env.sh run python -c "import lightgbm; print(lightgbm.__version__)"
```

**3. Jupyter.** Notebook'un IceTray'i görmesinin tek yolu, kernel'in
env-shell içindeki python olması. En temizi Jupyter'i ortam içinden
başlatmak:

```bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
cd <build_dizini> && ./env-shell.sh
cd ~/l4/osncnextl4 && python -m jupyter lab --no-browser --port=8896
```

Başlatılmadıysa `./setup_env.sh kernel` ile kernel kaydedilip notebook'ta
seçilir.

> **Tuzak:** Jupyter'in çalışma dizini **kernel restart ile değişmez**,
> sunucudan gelir. `returncode=2` ve `~/.local/share/Trash/...` gibi bir yol
> görüyorsan sunucu yanlış dizinde; kapatıp doğru dizinden yeniden başlat.
> Notebook'ta kontrol: `import os; os.getcwd()`.

## Kullanım

Her şey `oscNext_L4.ipynb` içinden. Bölümler:

| # | ne yapar |
|---|---|
| 0 | konfigürasyon + ortam kontrolü |
| 1 | L3 → L4 işleme (`process_L4.py`, + smoke test) |
| 2 | booking doğrulaması — HDF5'te gerçekte ne var |
| 3 | feature registry + `FEATURE_MAP` ↔ `REGISTRY` tutarlılığı |
| 4 | HDF5 → numpy |
| 5 | ağırlıklar (`w_phys`) |
| 6 | `.npz` eğitim setleri + train/test ayrımı |
| 7 | eğitim (`train_L4_classifier.py`) |
| 8 | doğrulama — verim/red tablosu, overtraining açığı, grafikler |
| 9 | kesim seçimi |
| 10 | modeli frame'e uygulama |

Komut satırından da çalışır:

```bash
# isleme
python process_L4.py --input-list nue_good.txt --output L4_output/hdf5/nue/L4_nue.hdf5 \
    --gcd <GCD> --scan off

# egitim
python train_L4_classifier.py --tag noise \
    --dataset L4_output/ds/L4_noise_dataset.npz --outdir L4_output/models
```

## Bilinen tuzaklar

**Bozuk girdi dosyaları.** pass3 üretiminde yarım yazılmış `.i3.zst`'ler
var; `I3Reader` listeyi tek seferde aldığı için bir bozuk dosya **tüm
tray'i öldürür**. `process_L4.py` iki katmanlı korur: ön tarama
(`--scan quick`, varsayılan) ve çalışma anı yeniden deneme
(`--retries 3`). Elenenler `<çıktı>.hdf5.badfiles.txt`'ye yazılır. Set
başına bir kez `scan_files.py --good-list` çalıştırıp `--scan off`
kullanmak en verimlisi.

**`--n` frame sayar, olay saymaz.** `--n 200` ile 60 olay book edilmesi
normal: akışta G/C/D/Q/P frame'leri var, P'lerin bir kısmı
`--sub-event-stream`'e uyar, bir kısmı L3 kesimini geçer. Çıktı kademeyi
gösterir.

**Modül önbelleği.** `git pull` sonrası `cannot import name ... from
l4_data` alıyorsan modül hafızada eski. Notebook bölüm 0 `%autoreload 2`
açıyor, tekrarlamamalı; olursa Kernel → Restart.

**Hızlandırma otomatik değil.** Varsayılan tek süreç. Paralellik için
`run_all(jobs=8, chunk_files=10, skip_optional=True)`. cobalt paylaşılan
makine — `jobs=8` makul, `jobs=64` değil.

## Notebook'u commit'lemeden önce

```bash
pip install --user nbstripout && nbstripout --install
```

Çıktı hücreleri MB'larca yer kaplar ve anlamsız diff üretir.

## Durum

- [x] Ortam doğrulandı, IceTray/lightgbm import katmanı
- [x] Bozuk girdi dosyalarına dayanıklılık
- [x] Sütun isimleri kesinleştirildi (14/14 BDT girdisi bulundu)
- [x] νe ve CORSIKA işlendi
- [x] noise sınıflandırıcısı eğitildi — %99 redde %95.9 verim
      (Tablo 13: ~%96)
- [ ] νμ / noise yeniden işlenmeli (bozuk `.i3.zst` yüzünden yarım kaldı)
- [ ] muon sınıflandırıcısı eğitilmedi
- [ ] Yeniden yazılan değişkenler (VICH, accumulated_time) referansla
      doğrulanmadı
- [ ] Gürültü MC istatistiği yetersiz — %99'un sağı ölçülemiyor

Ayrıntı ve açık riskler: `CLAUDE.md`. Teknik notla satır satır
karşılaştırma: `TEKNIK_NOT_KARSILASTIRMA.md`. Akış: `AKIS_SEMASI.md`.
