#!/usr/bin/env python
'''
Girdi .i3 dosyalarini tarayip bozuk olanlari bul.

process_L4.py zaten --scan ile bunu kendisi yapiyor; bu script ayni taramayi
tek basina, isleme baslamadan calistirmak icin.  Uretimden ONCE bir kez
calistirip kara listeyi olusturmak en verimlisi: her job kendi taramasini
tekrar etmez.

Kullanim:

  # Hizli tarama (dosya basina ilk 25 frame) -- kesik dosyalari yakalar
  python scan_files.py '/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst'

  # Tam tarama (her frame okunur, yavas ama kesin)
  python scan_files.py --full '/data/.../*.i3.zst'

  # Saglam dosyalarin listesini yaz -> process_L4.py --input-list ile kullan
  python scan_files.py --good-list good_23799.txt '/data/.../*.i3.zst'

Cikti:
  <goodlist>          saglam dosyalar, satir basina bir yol
  scan_bad.txt        bozuk dosyalar + sebep  (--bad-list ile degistirilir)
'''

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icetray_env import require_icetray, IceTrayNotAvailable

try:
    require_icetray()
except IceTrayNotAvailable as _e:
    sys.exit("\n" + str(_e) + "\n")

from process_L4 import validate_files


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="+", help="Dosya yollari ya da glob desenleri")
    p.add_argument("--full", action="store_true",
                   help="Her frame'i oku (yavas ama kesin).  Varsayilan: ilk N frame.")
    p.add_argument("--frames", type=int, default=25,
                   help="Hizli modda dosya basina okunacak frame (varsayilan 25)")
    p.add_argument("--good-list", default=None, help="Saglam dosyalari buraya yaz")
    p.add_argument("--bad-list", default="scan_bad.txt", help="Bozuk dosyalari buraya yaz")
    args = p.parse_args()

    files = []
    for pattern in args.input:
        m = sorted(glob.glob(pattern))
        files.extend(m if m else [pattern])
    if not files:
        sys.exit("Dosya bulunamadi.")

    print("Taranacak dosya: %d  (%s mod)"
          % (len(files), "tam" if args.full else "hizli/%d frame" % args.frames))

    good, bad = validate_files(files, n_frames=0 if args.full else args.frames)

    print()
    print("=" * 62)
    print("Saglam : %d" % len(good))
    print("Bozuk  : %d" % len(bad))
    print("=" * 62)

    if bad:
        print()
        for path, why in bad:
            print("  %s" % path)
            print("      %s" % why)
        with open(args.bad_list, "w") as fh:
            for path, why in bad:
                fh.write("%s\t%s\n" % (path, why))
        print("\n-> %s" % args.bad_list)

    if args.good_list:
        with open(args.good_list, "w") as fh:
            fh.write("\n".join(good) + "\n")
        print("-> %s  (%d dosya)" % (args.good_list, len(good)))
        print("\nKullanim:")
        print("  python process_L4.py --input-list %s --scan off ..." % args.good_list)

    # Bozuk dosya varsa cikis kodu 1 -- script'ten kontrol edilebilsin
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
