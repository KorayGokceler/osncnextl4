# Notebook hücreleri

Notebook `.gitignore`'da — yerel kopya asıl kopya. Mantık `.py` modüllerinde
durduğu için notebook'ta sadece aşağıdaki kısa çağrılar kalıyor. Modüller
`git pull` ile güncellenir; bu hücreleri bir kez yapıştırman yeter.

> Değiştirdikten sonra **Kernel → Restart Kernel**, sonra hücreleri baştan
> sırayla çalıştır. Notebook'ta hücreyi düzenlemek hafızadaki eski tanımı
> değiştirmez.

---

## Hücre 6 — `run_process` tanımı (tamamını sil, bunu koy)

```python
from l4_run import configure_runner, run_process, run_all
configure_runner(SAMPLES, PROCESS_PY, GCD)
```

`configure_runner` modüle `SAMPLES`, `PROCESS_PY`, `GCD` tanımlarını
veriyor — bunlar notebook'ta kalmaya devam ediyor (bölüm 1'deki hücre).

---

## Hücre 8 — smoke test (aynı, sadece bilgi)

```python
smoke = run_process("nue", n_frames=200)
```

`n_frames` verilince `--scan off` otomatik ekleniyor ve `--chunk-files`
kapanıyor (`--n` ile birlikte kullanılamaz).

---

## Hücre 10 — toplu işleme (yorumlu döngünün yerine)

```python
results = run_all(chunk_files=10)
```

`chunk_files=10` → her 10 L3 dosyası ayrı bir parça (`L4_nue_part000.hdf5`,
`_part001` …). Faydası: **gerçek yüzde ve ETA**, ve çökme halinde
tamamlanan parçalar atlanıyor — kaldığı yerden devam. `chunk_files=0`
tek parça (ilerleme belirsiz kalır, sadece sayaçlar akar).

---

## Hücre 12 — `dump_tables` (tamamını sil, bunu koy)

```python
from l4_data import dump_tables
TABLES = dump_tables(SAMPLES["nue"]["hdf5"].replace(".hdf5", "_smoke.hdf5"))
```

---

## Hücre 14 — registry (tamamını sil, bunu koy)

```python
from l4_data import (REGISTRY, AUX, NOISE_FEATURES, MUON_FEATURES, WANTED,
                     check_registry, check_feature_map)

print("--- noise ---");          check_registry(TABLES, NOISE_FEATURES)
print("--- muon ---");           check_registry(TABLES, MUON_FEATURES)
print("--- agirlik (AUX) ---");  check_registry(TABLES, list(AUX))
print()
check_feature_map()
```

`check_feature_map()` yeni: `REGISTRY` ile `l4_classifier_module.FEATURE_MAP`
çakışıyor mu? Eğitimde bir kolon, frame'e uygularken başka bir kolon
okunursa model **hata fırlatmadan** saçmalar. AST ile okuyor, icetray
gerekmiyor.

---

## Hücre 16 — yükleme (tamamını sil, bunu koy)

```python
from l4_data import load_sample

data = {}
for name in SAMPLES:
    d = load_sample(name, SAMPLES, WANTED)
    if d is not None:
        data[name] = d
```

---

## Hücre 20 — ağırlıklar (tamamını sil, bunu koy)

```python
from l4_data import add_weights
add_weights(data)
```

Çıktı artık teknik notun Tablo 13 değerleriyle karşılaştırıyor:

```
nue
  toplam oran = 9.4e-04 Hz  (0.940 mHz),  maks/toplam = 0.31%
      Tablo 13 (L3, pass2): 0.950 mHz  ->  bizim/nota oran = 0.99
```

Oran 0.1–10 aralığı dışındaysa `[!] MERTEBE SAPMASI` basıyor.

---

# Modüllerde ne var

| Dosya | İçerik |
|---|---|
| `l4_run.py` | `configure_runner`, `run_process`, `run_all`, ilerleme çubuğu |
| `l4_data.py` | `REGISTRY`, `ALTS`, `dump_tables`, `check_registry`, `check_feature_map`, `load_sample`, `add_weights` |

## Bu taşımada düzeltilen üç şey

**1. Ağırlık böleni 100 kat yanlıştı.** `_n_files` HDF5 dosya sayısıydı;
bölen L3 dosya sayısı olmalı. 100 L3 dosyası tek HDF5'e book edilince bölen
1 çıkıyordu. `process_L4.py` artık `<çıktı>.hdf5.meta.json` içine
`n_l3_files` yazıyor, `load_sample` onu topluyor. Sidecar'ı olmayan eski
HDF5'lerde uyarı basıyor — o setleri yeniden üretmek gerekiyor.

**2. `iLineFit_speed` kolon adı çakışıyordu.** `REGISTRY` `LFVel`,
`FEATURE_MAP` `lf_vel` diyordu; teknik not Tablo 11 ise değişkeni
`L4_iLineFit.speed` diye veriyor. Üçü de artık `ALTS`'te — dosyada
**gerçekten hangisi varsa** o kullanılıyor.

**3. Çözülemeyen değişken `KeyError` veriyordu.** Artık NaN ile dolduruluyor
ve örnek başına bir kez uyarı basılıyor.
