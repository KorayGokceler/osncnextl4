'''
oscNext Level 4 degisken hesaplama -- calisir hale getirilmis surum.

Orijinal tray segment'i (oscNext_level4.py) tamamen yoruma alinmisti ve su uc
eski projeye bagimliydi:

    tau_bdt.I3CutL7Module        -> VICH degiskenleri
    analysis.event_selection     -> Dunkman degiskenleri
    slc-veto SmallQ_Box          -> QR box

Bunlar modern IceTray meta-projelerinde genelde bulunmuyor. Burada VICH ve
Dunkman degiskenleri technical note'taki tanimlarina gore saf Python'la
yeniden yazildi; QR box opsiyonel birakildi (BDT girdisi degil).

Referans: oscNext technical note v00.07, bolum 3.4-3.6, Tablo 11-12.
'''

import os
import sys
import numpy as np

# icetray_env bu dosyanin yanindadir; baska bir dizinden import edildiginde
# de bulunabilmesi icin sys.path'e ekleniyor.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icetray_env import (require_icetray, optional_project, require_project,
                         load_deserialization_libs, deepcore_doms,
                         deepcore_veto_domset, deepcore_fiducial_domset,
                         load_lib)

require_icetray()
from icecube import dataclasses, icetray

# --- Opsiyonel projeler -----------------------------------------------------
# Modul seviyesinde SERT import etmiyoruz: tek bir eksik proje (orn. kendi
# derlediginiz build'de tensor_of_inertia yoksa) tum repoyu import edilemez
# hale getirmesin.  Eksik olan, o degiskeni ureten segment cagrildiginda
# net bir hata verir; digerleri calismaya devam eder.
DomTools          = optional_project("DomTools")
linefit           = optional_project("linefit")
tensor_of_inertia = optional_project("tensor_of_inertia")
fill_ratio        = optional_project("fill_ratio")

# Deserialization icin gerekli (dogrudan kullanilmasalar da)
load_deserialization_libs()


# ---------------------------------------------------------------------------
# Cikti frame objesi isimleri  (orijinal script ile birebir ayni)
# ---------------------------------------------------------------------------

L4_CUT_BOOL_KEY = "L4_Cut_Bool"

L4_FIRST_HLC_KEY      = "L4_first_hlc"
L4_FIRST_HLC_RHO_KEY  = L4_FIRST_HLC_KEY + "_rho"

L4_TOI_KEY            = "L4_ToI"
L4_LINEFIT_KEY        = "L4_iLineFit"
L4_QRBOX_KEY          = "L4_QR_Box"

L4_VICH_NCH_KEY       = "L4_VICH_nch"
L4_VICH_NPULSES_KEY   = "L4_VICH_npulses"
L4_VICH_QTOT_KEY      = "L4_VICH_qtot"

L4_SEP_IN_COGS_KEY    = "L4_separation_in_cogs"
L4_ACC_TIME_KEY       = "L4_accumulated_time"

L4_FTLR_KEY           = "L4_FullTimeLengthRatio"
L4_NFLUX_KEY          = "L4_n_flux_events"   # I3GenieInfo'dan P frame'e tasinir
L4_MICROCOUNT_KEY     = "L4_micro_count"
L4_FILL_RATIO_KEY     = "L4_fill_ratio"

L4_NOISE_STRAIGHT_CUT_KEY         = "L4_NoiseStraightCuts_Bool"
L4_NOISE_MODEL_PREDICTION_KEY     = "L4_NoiseClassifier_ProbNu"
L4_MUON_MODEL_PREDICTION_DATA_KEY = "L4_MuonClassifier_Data_ProbNu"

# ---------------------------------------------------------------------------
# Pulse serisi isimleri
#
# oscNext L3 (online_filterscripts .. grecovariables.DeepCoreCleaning) girdi
# olarak "SplitInIcePulses" alir ve "SRTTWSplitInIcePulsesDC" uretir.
# (Eski pass1 isimlendirmesi "SRTTWOfflinePulsesDC" idi -- ARTIK O DEGIL.)
# ---------------------------------------------------------------------------
UNCLEANED_PULSES_DEFAULT = "SplitInIcePulses"
CLEANED_PULSES_DEFAULT   = "SRTTWSplitInIcePulsesDC"

# HitStatistics / HitMultiplicity (muon BDT'nin cog_z, z_sigma, z_travel girdileri)
#
# NOT: L3 bunlari hesaplayip sonra SILIYOR (cleanup listesindeki
# "<name>_DeepCoreCutsSRTTWSplitInIcePulsesDCHitStatistics").  Bu yuzden
# oscNext_L4_hit_statistics segment'i ile YENIDEN hesapliyoruz -- ayni pulse
# serisi, ayni modul, dolayisiyla ayni sonuc.
HITSTAT_KEY  = CLEANED_PULSES_DEFAULT + "HitStatistics"
HITMULT_KEY  = CLEANED_PULSES_DEFAULT + "HitMultiplicity"

# Micro count icin kullanilan static/dynamic time window parametreleri
STW_MINUS = 3500.
STW_PLUS  = 4000.
DTW       = 200
# Orijinal script'teki format string birebir korundu ->  "STW_m3500p4000_DTW200"
# (Eski GRECO kodu ayni degisken icin "STW7500_DTW200" ariyordu -- uyusmuyor.)
MICROCOUNT_SUBKEY = "STW_m%ip%i_DTW%i" % (STW_MINUS, STW_PLUS, DTW)

# Fill ratio yaricap carpani -- GRECO icin optimize edildi, oscNext icin
# yeniden optimize EDILMEDI.  Yeni veride oynanabilecek bir parametre.
FILL_RATIO_SPHERICAL_RADIUS_MEAN = 1.6

# VICH nedensellik hiz penceresi [m/ns] (technical note bolum 3.4)
VICH_SPEED_MIN = 0.25
VICH_SPEED_MAX = 0.40

# String 36 konumu (DeepCore merkezi) -- calc_rho_36 icin fallback
STRING36_X = 46.29
STRING36_Y = -34.88


L4_HDF5_KEYS = [
    L4_CUT_BOOL_KEY,
    L4_FIRST_HLC_KEY, L4_FIRST_HLC_RHO_KEY,
    L4_TOI_KEY, L4_TOI_KEY + "Params",
    L4_LINEFIT_KEY, L4_LINEFIT_KEY + "Params",
    L4_QRBOX_KEY,
    L4_VICH_NCH_KEY, L4_VICH_NPULSES_KEY, L4_VICH_QTOT_KEY,
    L4_SEP_IN_COGS_KEY, L4_ACC_TIME_KEY,
    L4_MICROCOUNT_KEY, L4_FILL_RATIO_KEY, L4_FTLR_KEY, L4_NFLUX_KEY,
    L4_NOISE_STRAIGHT_CUT_KEY,
    L4_NOISE_MODEL_PREDICTION_KEY,
    L4_MUON_MODEL_PREDICTION_DATA_KEY,
]


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

_LC_FLAG = dataclasses.I3RecoPulse.PulseFlags.LC


def iter_map(pulse_map):
    '''
    (omkey, pulses) ciftleri uzerinde iterasyon.

    DIKKAT: IceTray surumune gore bir I3RecoPulseSeriesMap uzerinde dogrudan
    iterasyon ANAHTARLARI dondurebilir, cift degil.  O durumda
    "for omkey, pulses in pmap" hatasi verir -- OMKey uc bilesene
    (string, om, pmt) acildigi icin "too many values to unpack (expected 2)".
    .items() her surumde dogru calisir.
    '''
    if pulse_map is None:
        return []
    try:
        return pulse_map.items()
    except AttributeError:
        return iter(pulse_map)


def get_pulses(frame, key):
    '''Pulse serisini al; mask/union ise frame'e uygula.'''
    if key not in frame:
        return None
    obj = frame[key]
    if hasattr(obj, "apply"):
        try:
            return obj.apply(frame)
        except Exception:
            return None
    return obj


def calc_rho_36(x, y):
    '''String 36'ya (DeepCore merkezi) yatay radyal uzaklik.'''
    return float(np.hypot(x - STRING36_X, y - STRING36_Y))


def iter_hits(pulse_map, geometry, first_pulse_only=False):
    '''
    (omkey, pos, time, charge) uzerinde iterasyon.
    first_pulse_only=True ise DOM basina sadece ilk pulse.
    '''
    omgeo = geometry.omgeo
    for omkey, pulses in iter_map(pulse_map):
        if omkey not in omgeo:
            continue
        pos = omgeo[omkey].position
        if first_pulse_only:
            if len(pulses):
                p = pulses[0]
                yield omkey, pos, p.time, p.charge
        else:
            for p in pulses:
                yield omkey, pos, p.time, p.charge


def charge_weighted_cog(hits):
    '''Yuk agirlikli center-of-gravity (konum + zaman).'''
    xs, ys, zs, ts, qs = [], [], [], [], []
    for _, pos, t, q in hits:
        xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
        ts.append(t); qs.append(max(q, 0.0))
    if not xs:
        return None
    q = np.asarray(qs, dtype=float)
    if q.sum() <= 0:
        q = np.ones_like(q)
    w = q / q.sum()
    return (float(np.dot(w, xs)), float(np.dot(w, ys)),
            float(np.dot(w, zs)), float(np.dot(w, ts)))


def check_object_exists(frame, object_key):
    '''Obje yoksa frame'i dusur (orijinaldeki data_quality.check_object_exists).'''
    return object_key in frame


# ===========================================================================
# 1. ORTAK DEGISKENLER
# ===========================================================================

def _first_hlc(frame, pulses_key, output_key, geometry_key="I3Geometry"):
    '''
    Temizlenmis seride zamanca ilk HLC hit'i bul, I3Particle olarak yaz.

    Orijinal script "FirstHLC<I3RecoPulse>" C++ modulunu kullaniyordu.  Bu
    modul her meta-projede bulunmadigi icin ayni isi Python'da yapiyoruz:
    HLC hit'ler I3RecoPulse.PulseFlags.LC bayragi ile isaretlenir.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True
    omgeo = frame[geometry_key].omgeo

    best = None   # (time, pos)
    for omkey, pulses in iter_map(pmap):
        if omkey not in omgeo:
            continue
        for p in pulses:
            if not (p.flags & _LC_FLAG):
                continue
            if best is None or p.time < best[0]:
                best = (p.time, omgeo[omkey].position)
            break   # DOM basina ilk HLC pulse yeterli

    if best is None:
        return True

    part = dataclasses.I3Particle()
    part.pos = best[1]
    part.time = best[0]
    part.shape = dataclasses.I3Particle.ParticleShape.Cascade
    part.fit_status = dataclasses.I3Particle.FitStatus.OK
    frame[output_key] = part
    return True


def _add_rho_36(frame, particle_key, output_key):
    if particle_key in frame and output_key not in frame:
        pos = frame[particle_key].pos
        frame[output_key] = dataclasses.I3Double(calc_rho_36(pos.x, pos.y))
    return True


def _full_time_length_ratio(frame, output_key,
                            l3_key="IC2018_LE_L3_Vars",
                            cleaned_pulses=None, uncleaned_pulses=None):
    '''
    Temizlenmis / temizlenmemis olay suresi orani.  Noise BDT girdisi.

    Technical note Tablo 11: "olayin toplam suresini hem temizlenmis hem
    temizlenmemis pulse serisinde olc (sure = maks pulse zamani - min pulse
    zamani), ikisinin oranini hesapla."

    Yon: temizlenmis / temizlenmemis, yani [0,1] araliginda (Sekil 13'teki
    x ekseni ile uyumlu).

    OLCULEN DAVRANIS (test_noise_vars.py, birer L3 dosyasi: 126 nue,
    17 noise) -- onceki yorumdaki fizik hikayesi YANLISTI, duzeltildi:

                       oran(medyan)   temizlenmis    temizlenmemis
        nue                  0.16        1626 ns        10100 ns
        noise (L3 gecen)     0.27        2780 ns        10290 ns

      * Oran hicbir zaman 1'e yaklasmiyor.  Temizlenmemis sure her olayda
        ~10 us -- cunku SplitInIcePulses tum okuma penceresini kapliyor ve
        gurultu hitleri her yerde.  Yani degisken fiilen
        "temizlenmis sure / 10 us".
      * Ayirt etme yonu beklediginin TERSI: gurultu olaylarinin temizlenmis
        serisi nue'ninkinden daha UZUN (dusuk enerjili kaskad zamanda
        kompakt; L3'u gecmis gurultu olayi zamanda dagilmis birkac hit).
        Ayirt ediyor, ama "gercek olay ~1 / gurultu ~0" degil.
        (17 noise olayi az -- siralama bu dosyada boyle, magnitud tespiti
        ise yapisal ve kesin.)

    pass3 L3 ciktisinda IC2018_LE_L3_Vars icinde CleanedFullTimeLength ve
    UncleanedFullTimeLength AYRI AYRI var ama ORANLARI YOK -- bu yuzden
    burada hesapliyoruz.  L3 map'inde yoksa pulse serilerinden dogrudan
    olculur.
    '''
    if output_key in frame:
        return True

    cleaned = uncleaned = None

    if l3_key in frame:
        v = frame[l3_key]
        if "CleanedFullTimeLength" in v and "UncleanedFullTimeLength" in v:
            cleaned = float(v["CleanedFullTimeLength"])
            uncleaned = float(v["UncleanedFullTimeLength"])

    if cleaned is None and cleaned_pulses and uncleaned_pulses:
        def duration(key):
            pmap = get_pulses(frame, key)
            if pmap is None:
                return None
            times = [p.time for _, pulses in iter_map(pmap) for p in pulses]
            return (max(times) - min(times)) if times else None
        cleaned = duration(cleaned_pulses)
        uncleaned = duration(uncleaned_pulses)

    if cleaned is None or uncleaned is None or uncleaned <= 0:
        return True

    # uncleaned <= 0 sifira bolmeyi engelliyor ama NaN'i ENGELLEMIYOR:
    # NaN <= 0 -> False, yani NaN bir payda bu kontrolden gecip sonuca
    # sessizce NaN yazardi.  Sonucu dogrudan denetle -- inf de NaN de elenir,
    # degisken yazilmaz ve HDF5'te eksik gorunur (yanlis sayidan iyidir).
    ratio = float(cleaned) / float(uncleaned)
    if not np.isfinite(ratio):
        return True

    # Temizlenmis seri temizlenmemisin alt kumesi oldugu icin oran <= 1
    # olmali.  Asiyorsa iki sureyi farkli temel serilerden olcuyoruz
    # demektir -- sessizce gecmesin.
    if ratio > 1.0 and not getattr(_full_time_length_ratio, "_warned", False):
        _full_time_length_ratio._warned = True
        icetray.logging.log_warn(
            "FullTimeLengthRatio > 1 (%.3f): temizlenmis (%.1f ns) sure "
            "temizlenmemisten (%.1f ns) uzun.  Iki sure ayni temel seriden "
            "olculmuyor olabilir." % (ratio, cleaned, uncleaned))

    frame[output_key] = dataclasses.I3Double(ratio)
    return True


@icetray.traysegment
def oscNext_L4_common_variables(tray, name, cleaned_pulses, uncleaned_pulses=None):
    '''Ilk HLC hit, rho, ve FullTimeLengthRatio.'''

    tray.Add(_first_hlc, name + "_FirstHLC",
             pulses_key=cleaned_pulses,
             output_key=L4_FIRST_HLC_KEY)

    tray.Add(_add_rho_36, name + "_FirstHLCRho",
             particle_key=L4_FIRST_HLC_KEY,
             output_key=L4_FIRST_HLC_RHO_KEY)

    tray.Add(_full_time_length_ratio, name + "_FTLR",
             output_key=L4_FTLR_KEY,
             cleaned_pulses=cleaned_pulses,
             uncleaned_pulses=uncleaned_pulses)


class PropagateGenieInfo(icetray.I3Module):
    '''
    I3GenieInfo.n_flux_events'i her Physics frame'e I3Double olarak yazar.

    NEDEN: I3GenieInfo dosya basina bir kez, Physics OLMAYAN bir frame'de
    (S/M) bulunur.  HDF5 booking ise olay bazlidir -- o yuzden degeri her
    P frame'e tasimazsak agirlik hesabinda kullanamayiz.

    Agirlik konvansiyonu (mevcut oscnext_rates.py ile ayni):
        weight [Hz] = OneWeight * flux(E) / n_flux
        n_flux      = I3GenieInfo.n_flux_events            (varsa)
                    = NEvents * 0.7 (nu) veya 0.3 (nubar)  (yoksa)
    '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter("OutputKey", "Yazilacak anahtar", L4_NFLUX_KEY)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.output_key = self.GetParameter("OutputKey")
        self.n_flux = None
        self.warned = False
        self.n_seen = 0          # kac I3GenieInfo goruldu (= kac L3 dosyasi)

    def _grab(self, frame):
        '''
        I3GenieInfo'yu okumayi dene.  Deserialization hatasi (genie_icetray
        import edilmemis) ya da eksik alan job'i COKURMEMELI -- agirlik
        hesabi NEvents fallback'ine duser.

        DIKKAT: her L3 DOSYASININ kendi I3GenieInfo'su var ve bir tray
        birden fazla dosya isliyor (--chunk-files).  Deger her yeni
        I3GenieInfo'da GUNCELLENMELI; onceden bir kez okunup sabitleniyordu,
        yani ilk dosyanin n_flux_events'i sonraki dosyalarin olaylarina da
        uygulaniyordu ve o olaylarin agirligi yanlis cikiyordu.
        '''
        if not frame.Has("I3GenieInfo"):
            return
        try:
            new_val = float(frame["I3GenieInfo"].n_flux_events)
            self.n_seen += 1
            if self.n_flux is not None and new_val != self.n_flux:
                icetray.logging.log_info(
                    "PropagateGenieInfo: n_flux_events degisti %g -> %g "
                    "(dosya %d) -- guncelleniyor"
                    % (self.n_flux, new_val, self.n_seen))
            self.n_flux = new_val
            icetray.logging.log_info(
                "PropagateGenieInfo: n_flux_events = %g" % self.n_flux)
        except Exception as e:
            if not self.warned:
                icetray.logging.log_warn(
                    "PropagateGenieInfo: I3GenieInfo okunamadi (%s: %s). "
                    "genie_icetray import edildi mi? Agirlik hesabi "
                    "NEvents * nu/nubar fraksiyonuna dusecek."
                    % (type(e).__name__, e))
                self.warned = True

    # NOT: Process() override edildigi icin DAQ()/Simulation() gibi
    # stream metotlari CAGRILMAZ -- dagitimi asagidaki Process yapiyor ve
    # her stream'de _grab cagiriyor.  (Eskiden ikisi de vardi; DAQ ve
    # Simulation olu koddu.)
    def Process(self):
        frame = self.PopFrame()
        if frame.Stop != icetray.I3Frame.Physics:
            self._grab(frame)
            self.PushFrame(frame)
            return
        self._grab(frame)
        if self.n_flux is not None and self.output_key not in frame:
            frame[self.output_key] = dataclasses.I3Double(self.n_flux)
        elif self.n_flux is None and not self.warned:
            icetray.logging.log_warn(
                "PropagateGenieInfo: I3GenieInfo bulunamadi -- agirlik hesabi "
                "NEvents * nu/nubar fraksiyonuna dusecek")
            self.warned = True
        self.PushFrame(frame)


# ===========================================================================
# 2. MUON REDDI DEGISKENLERI
# ===========================================================================

def _accumulated_time(frame, pulses_key, output_key, fraction=0.75):
    '''
    Olayin toplam yukunun %75'ine ulasmasi icin gecen sure [ns].

    Technical note Tablo 12: "Time to reach 75% of an event's charge in the
    cleaned pulse series."  Seciimdeki nadir yuke-bagli degiskenlerden biri.
    Orijinalde analysis/event_selection'in CalculateVariables modulunden
    cikariliyordu; burada dogrudan hesapliyoruz.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None:
        return True

    times, charges = [], []
    for _, pulses in iter_map(pmap):
        for p in pulses:
            times.append(p.time)
            charges.append(max(p.charge, 0.0))
    if not times:
        return True

    t = np.asarray(times); q = np.asarray(charges)
    order = np.argsort(t)
    t, q = t[order], q[order]
    total = q.sum()
    if total <= 0:
        return True
    cum = np.cumsum(q) / total
    idx = int(np.searchsorted(cum, fraction))
    idx = min(idx, len(t) - 1)
    frame[output_key] = dataclasses.I3Double(float(t[idx] - t[0]))
    return True


def _separation_in_cogs(frame, pulses_key, output_key, geometry_key="I3Geometry"):
    '''
    Olayi zamanca iki yariya bolup her yarinin yuk agirlikli COG'unu hesapla,
    aradaki mesafeyi yaz.

    Track ilerledikce COG kayar -> buyuk ayrim.
    Cascade yerinde patlar    -> kucuk ayrim.

    DIKKAT: orijinal Dunkman implementasyonunun tam tanimi dogrulanamadi
    (proje mevcut degil).  Referans dosyalariniz varsa bu degiskeni onlarla
    karsilastirin.  BDT girdisi olmadigi icin kritik degil.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True

    hits = list(iter_hits(pmap, frame[geometry_key]))
    if len(hits) < 4:
        return True
    hits.sort(key=lambda h: h[2])          # zamana gore sirala
    half = len(hits) // 2
    c1 = charge_weighted_cog(hits[:half])
    c2 = charge_weighted_cog(hits[half:])
    if c1 is None or c2 is None:
        return True
    d = np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2 + (c2[2]-c1[2])**2)
    frame[output_key] = dataclasses.I3Double(float(d))
    return True


def _vich(frame, uncleaned_pulses, cleaned_pulses,
          nch_key, npulses_key, qtot_key, geometry_key="I3Geometry",
          fiducial_cog=True):
    '''
    Veto Identified Causal Hits.

    Technical note bolum 3.4: veto bolgesindeki her hit ile olayin COG
    konum/zaman vertex'i arasindaki hiz hesaplanir; hiz [0.25, 0.4] m/ns
    araligindaysa hit, detektoru gecen bir muondan kaynaklanmis olabilecegi
    gerekcesiyle isaretlenir.  (0.3 m/ns = isik hizi.)

    ONEMLI: girdi TEMIZLENMEMIS seri olmali -- temizleme muonun veto
    bolgesindeki zayif, izole hit'lerini siler ve muon tam da o hit'lerden
    taninir.  COG vertex'i ise TEMIZLENMIS seriden alinir.

    COG KAPSAMI (fiducial_cog):
      Teknik not §3.4 aynen: "the center-of-gravity (COG) of the hits
      INSIDE THE FIDUCIAL VOLUME is calculated".  Yani COG sadece fiducial
      DOM'lardan hesaplanmali.  Ilk yazimda tum temizlenmis seri
      kullaniliyordu; muonlu olaylarda veto hitleri COG'u yukari/disa
      cekiyor, dolayisiyla d ve t_COG kayiyor ve hiz penceresine dusme
      olasiligi degisiyor -- tam da ayirt etme gucunu bozacak yonde.

      fiducial_cog=True  -> nota uygun (VARSAYILAN)
      fiducial_cog=False -> eski davranis (karsilastirma icin)

      Not COG'un yuk agirlikli olup olmadigini SOYLEMIYOR; biz yuk
      agirlikli aliyoruz.  Bu hala dogrulanmamis bir varsayim.

    Orijinalde tau_bdt.I3CutL7Module yapiyordu.
    '''
    if nch_key in frame:
        return True
    unc = get_pulses(frame, uncleaned_pulses)
    cln = get_pulses(frame, cleaned_pulses)
    if unc is None or cln is None or geometry_key not in frame:
        return True

    geo = frame[geometry_key]

    # COG: teknik not §3.4 -> sadece FIDUCIAL hacimdeki hitler
    if fiducial_cog:
        fid = deepcore_fiducial_domset("IC86")
        cog = charge_weighted_cog(h for h in iter_hits(cln, geo) if h[0] in fid)
        if cog is None:
            # Fiducial'da hic hit yoksa geri dus -- olay zaten atilacak ama
            # sessizce yanlis sayi uretmektense tum seriyi kullan.
            cog = charge_weighted_cog(iter_hits(cln, geo))
    else:
        cog = charge_weighted_cog(iter_hits(cln, geo))

    if cog is None:
        return True
    cx, cy, cz, ct = cog

    # NOT: eskiden burada her olayda DOMS.DOMS("IC86") yeniden kuruluyordu.
    # Artik cache'li (icetray_env.deepcore_veto_domset).
    veto_doms = deepcore_veto_domset("IC86")

    n_doms, n_pulses, qtot = 0, 0, 0.0
    for omkey, pulses in iter_map(unc):
        if omkey not in veto_doms:
            continue
        if omkey not in geo.omgeo:
            continue
        pos = geo.omgeo[omkey].position
        d = np.sqrt((pos.x-cx)**2 + (pos.y-cy)**2 + (pos.z-cz)**2)
        dom_counted = False
        for p in pulses:
            dt = ct - p.time            # veto hit COG'dan ONCE olmali
            if dt <= 0:
                continue
            speed = d / dt
            if VICH_SPEED_MIN <= speed <= VICH_SPEED_MAX:
                n_pulses += 1
                qtot += max(p.charge, 0.0)
                dom_counted = True
        if dom_counted:
            n_doms += 1

    frame[nch_key]     = dataclasses.I3Double(float(n_doms))
    frame[npulses_key] = dataclasses.I3Double(float(n_pulses))
    frame[qtot_key]    = dataclasses.I3Double(float(qtot))
    return True


@icetray.traysegment
def oscNext_L4_atm_muon_classifier_variables(tray, name,
                                             uncleaned_pulses,
                                             cleaned_pulses,
                                             run_qr_box=False,
                                             run_optional=True):
    '''
    L4 atmosferik muon reddi siniflandiricisinin girdileri.

    run_optional=False: BDT girdisi OLMAYAN hesaplar atlanir
    (I3TensorOfInertia ve separation_in_cogs).  Ikisi de Tablo 12'de yok;
    aday/legacy olarak duruyorlar ama olay basina gercek maliyetleri var.
    Uretim hizlandirmak icin process_L4.py --skip-optional ile kapatilir.
    '''

    # Bu segment'in gerektirdigi projeler -- yoksa BURADA net hata ver
    # (import zamaninda degil, ki geri kalan repo import edilebilsin).
    require_project("linefit")

    # --- Tensor of inertia (BDT girdisi degil; aday/legacy) ---
    if run_optional:
        require_project("tensor_of_inertia")
        tray.AddModule("I3TensorOfInertia", name + "_ToI",
                       AmplitudeOption=1,
                       AmplitudeWeight=1,
                       InputReadout=cleaned_pulses,
                       InputSelection="",
                       MinHits=3,
                       Name=L4_TOI_KEY)

    # --- improved LineFit ---
    # BDT'de kullanilan alan hiz: L4_iLineFitParams.LFVel
    tray.AddSegment(linefit.simple, name + "_iLineFit",
                    inputResponse=cleaned_pulses,
                    fitName=L4_LINEFIT_KEY)

    # --- QR box (slc-veto; opsiyonel, BDT girdisi degil) ---
    if run_qr_box:
        try:
            if not load_lib("slc-veto"):
                raise RuntimeError("slc-veto kutuphanesi bu build'de yok")
            tray.AddModule("SmallQ_Box", name + "_QRBox",
                           BoxName=L4_QRBOX_KEY,
                           RecoPulsesKey=cleaned_pulses)
        except Exception as e:
            icetray.logging.log_warn("QR box atlandi (slc-veto yok): %s" % e)

    # --- Dunkman degiskenleri (Python yeniden yazim) ---
    tray.Add(_accumulated_time, name + "_AccTime",
             pulses_key=cleaned_pulses,
             output_key=L4_ACC_TIME_KEY)

    # BDT girdisi degil (Tablo 12'de yok) -- saf Python, olay basina maliyeti var
    if run_optional:
        tray.Add(_separation_in_cogs, name + "_SepCOG",
                 pulses_key=cleaned_pulses,
                 output_key=L4_SEP_IN_COGS_KEY)

    # --- VICH (tau_bdt yeniden yazim) ---
    tray.Add(_vich, name + "_VICH",
             uncleaned_pulses=uncleaned_pulses,
             cleaned_pulses=cleaned_pulses,
             nch_key=L4_VICH_NCH_KEY,
             npulses_key=L4_VICH_NPULSES_KEY,
             qtot_key=L4_VICH_QTOT_KEY)


# ===========================================================================
# 3. GURULTU REDDI DEGISKENLERI
# ===========================================================================

def _micro_count(frame, pulses_key, output_key, subkey):
    '''Temizlenmis+pencereli seride hit alan DOM sayisini I3MapStringInt olarak yaz.'''
    values = dataclasses.I3MapStringInt()
    pmap = get_pulses(frame, pulses_key)
    # DOM sayisi: len(map) her surumde calisir, .keys() bazen liste degil
    # bir view/iterator dondurur.
    if pmap is None:
        n_doms = 0
    else:
        try:
            n_doms = len(pmap)
        except TypeError:
            n_doms = sum(1 for _ in iter_map(pmap))
    values[subkey] = int(n_doms)
    if output_key not in frame:
        frame[output_key] = values
    return True


@icetray.traysegment
def oscNext_L4_noise_cut_variables(tray, name,
                                   fill_ratio_vertex,
                                   cleaned_pulses,
                                   micro_count_pulses=None):
    '''
    L4 saf gurultu reddi siniflandiricisinin girdileri.

    micro_count_pulses: micro_count zincirinin BASLADIGI pulse serisi.
        None (varsayilan) -> cleaned_pulses, yani teknik notun Tablo 11'de
        dedigi gibi ("Start with the cleaned pulse series").
        Orijinal pass2 kodunu birebir tekrarlamak icin buraya temizlenmemis
        seri verilir -- bkz. CLAUDE.md acik risk 5b.  fill_ratio her iki
        durumda da cleaned_pulses kullanir (orijinalde de oyle).
    '''
    if micro_count_pulses is None:
        micro_count_pulses = cleaned_pulses

    #
    # Micro count
    #
    # Teknik not, Tablo 11 (s.36) -- BIREBIR tarif:
    #   "Start with the cleaned pulse series.  Look at pulses occurring within
    #    [-3.5 us, +4 us] from the trigger time.  Slide a time window of 200 ns
    #    that maximizes the number of triggered DOMs in it.  Get the number of
    #    triggered DOMs in that sliding time window"
    #
    # Zincir:  cleaned -> static TW [-3500,+4000] ns -> DeepCore fiducial
    #          -> 200 ns dinamik pencere -> DOM say
    #
    # DUZELTME (bkz. CLAUDE.md "Booking/okuma denetimi"):  Bu zincir eskiden
    # TEMIZLENMEMIS seriden basliyor ve araya bir I3SeededRTCleaning koyuyordu
    # -- ama o modulun ciktisi (L4_SRTTWPulses) HICBIR yerde okunmuyordu:
    # I3OMSelection girdi olarak SeededRT ciktisini degil StaticTWC ciktisini
    # aliyordu.  Yani zincirdeki tek gurultu temizleme adimi fiilen devre disiydi
    # ve micro_count ham (temizlenmemis) hitler uzerinden sayiliyordu.  Gurultu
    # siniflandiricisinin girdisi icin bu dogrudan zararli: gurultu hitleri de
    # sayiliyor, degiskenin ayirt etme gucu dusuyordu.
    #
    # Artik notun dedigi gibi TEMIZLENMIS seriyle (SRTTWSplitInIcePulsesDC)
    # basliyoruz.  Ayrica SeededRT blogu kaldirildi: L3 bu seriye zaten SRT
    # temizligi uygulamis (serinin adindaki "SRT" bu), tekrar uygulamak cift
    # temizleme olurdu.
    #
    # L3'un kendi microcount'u (STW9000_DTW300Hits, [-4,+5] us / 300 ns) farkli
    # parametrelerde -- ikisi tam korele degil, BDT ikisinden de bilgi cikarir.

    require_project("DomTools")
    require_project("fill_ratio")
    if not load_lib("static-twc"):
        raise RuntimeError(
            "C++ kutuphanesi 'static-twc' bu build'de yok -- micro_count "
            "hesaplanamaz.  DomTools/static-twc derlenmis mi kontrol edin "
            "(python diagnose_env.py).")

    tw_pulses = "L4_TWPulses"
    tray.AddModule("I3StaticTWC<I3RecoPulseSeries>", name + "_StaticTWC_DC",
                   InputResponse=micro_count_pulses,
                   OutputResponse=tw_pulses,
                   TriggerConfigIDs=[1010, 1011],
                   TriggerName="I3TriggerHierarchy",
                   WindowMinus=STW_MINUS,
                   WindowPlus=STW_PLUS)

    # Klasik IC86 DeepCore fiducial hacmi
    dom_list = deepcore_doms("IC86")
    tw_fid_pulses = tw_pulses + "_DCFid"

    tray.AddModule("I3OMSelection<I3RecoPulseSeries>", name + "_DCFidPulses",
                   selectInverse=True,
                   InputResponse=tw_pulses,
                   OutputResponse=tw_fid_pulses,
                   OmittedKeys=dom_list.DeepCoreFiducialDOMs)

    dtw_pulses = tw_fid_pulses + ("_DTW%i" % DTW)
    tray.AddModule("I3TimeWindowCleaning<I3RecoPulse>", name + "_DynamicTW",
                   InputResponse=tw_fid_pulses,
                   OutputResponse=dtw_pulses,
                   TimeWindow=DTW)

    tray.Add(_micro_count, name + "_MicroCount",
             pulses_key=dtw_pulses,
             output_key=L4_MICROCOUNT_KEY,
             subkey=MICROCOUNT_SUBKEY)

    #
    # Fill ratio
    #
    # GRECO farkli bir pulse serisi kullaniyordu; burada standart temizlenmis
    # seri kullaniliyor (orijinal yorumda "iyi calisiyor, basitlik icin boyle
    # birakiyorum" deniyor).

    tray.AddModule("I3FillRatioModule", name + "_FillRatio",
                   RecoPulseName=cleaned_pulses,
                   ResultName=L4_FILL_RATIO_KEY,
                   SphericalRadiusMean=FILL_RATIO_SPHERICAL_RADIUS_MEAN,
                   VertexName=fill_ratio_vertex)


# ===========================================================================
# 4. HIT STATISTICS  (muon BDT'nin cog_z / z_sigma / z_travel girdileri)
# ===========================================================================

@icetray.traysegment
def oscNext_L4_hit_statistics(tray, name, cleaned_pulses):
    '''
    common_variables ile hit statistics ve multiplicity hesapla.
    L3'te zaten hesaplaniyorsa bu segment atlanabilir.
    '''
    require_project("common_variables")
    from icecube.common_variables import hit_statistics, hit_multiplicity

    tray.AddSegment(hit_statistics.I3HitStatisticsCalculatorSegment,
                    name + "_HitStatistics",
                    PulseSeriesMapName=cleaned_pulses,
                    OutputI3HitStatisticsValuesName=HITSTAT_KEY,
                    BookIt=False,
                    If=lambda f: HITSTAT_KEY not in f)

    tray.AddSegment(hit_multiplicity.I3HitMultiplicityCalculatorSegment,
                    name + "_HitMultiplicity",
                    PulseSeriesMapName=cleaned_pulses,
                    OutputI3HitMultiplicityValuesName=HITMULT_KEY,
                    BookIt=False,
                    If=lambda f: HITMULT_KEY not in f)


# ===========================================================================
# 5. KESIMLER
# ===========================================================================

def L4_noise_straight_cuts(frame, output_key=L4_NOISE_STRAIGHT_CUT_KEY):
    '''
    Siniflandiriciya alternatif gevsek duz kesimler.  Uretiliyor ama kesimde
    kullanilmiyor -- siniflandirici ile sorun cikarsa diye tutuluyor.
    Her degiskenin hangi yonde ayirdigini gormek icin de faydali bir referans.
    '''
    try:
        keep = (
            frame[HITMULT_KEY].n_hit_doms >= 8 and
            frame["IC2018_LE_L3_Vars"]["STW9000_DTW300Hits"] >= 2 and
            frame[L4_MICROCOUNT_KEY][MICROCOUNT_SUBKEY] >= 2 and
            frame[L4_FILL_RATIO_KEY].fill_ratio_from_mean >= 0.03 and
            frame[HITSTAT_KEY].z_sigma >= 8. and
            frame[HITSTAT_KEY].z_travel >= -50.
        )
    except (KeyError, AttributeError):
        keep = False
    frame[output_key] = icetray.I3Bool(bool(keep))
    return True


@icetray.traysegment
def compute_L4_cut(tray, name, classifier_model_dir,
                   noise_cut=0.70, muon_cut=0.65):
    '''
    Egitilmis siniflandiricilari uygula ve L4 kesimini hesapla.

    NOT: oscNext projesi (icecube.oscNext.tools.classifier.I3Classifier) bu
    meta-projede YOK.  Yerine l4_classifier_module.py kullaniliyor -- ayni isi
    yapar, sadece lightgbm + numpy'a bagimlidir (IceTray ortaminda sklearn ve
    joblib bulunmuyor).

    Modeller HENUZ EGITILMEMISSE bu segment'i atlayin: once degiskenleri
    kesimsiz book edip siniflandiricilari egitmeniz gerekir.
    '''
    from l4_classifier_module import add_L4_classifiers

    tray.Add(L4_noise_straight_cuts, name + "_straight_cuts")

    tray.Add(add_L4_classifiers, name + "_classifiers",
             model_dir=classifier_model_dir,
             noise_cut=noise_cut,
             muon_cut=muon_cut,
             apply_cut=True)


# ===========================================================================
# 6. ANA SEGMENT
# ===========================================================================

@icetray.traysegment
def oscNext_L4(tray, name,
               uncleaned_pulses=UNCLEANED_PULSES_DEFAULT,
               cleaned_pulses=CLEANED_PULSES_DEFAULT,
               apply_l3_cut=True,
               is_genie=False,
               compute_hit_statistics=True,
               run_optional=True,
               apply_cut=False,
               classifier_model_dir=None,
               micro_count_uncleaned=False):
    '''
    oscNext L4 ana tray segment'i.

    apply_cut=False (varsayilan): sadece degiskenleri hesaplar.  Modelleri
    egitmeden once bu modda calistirin -- kesim uygulanmadan tum olaylari
    book edersiniz, boylece hem noise hem muon egitim setini tek gecisten
    cikarabilirsiniz.

    micro_count_uncleaned=True: micro_count zinciri temizlenmemis seriden
    baslar -- orijinal pass2 kodunun (hatali) davranisi.  Sadece pass2
    sayilarini/Sekil 12'yi tekrarlamak icin.  Varsayilan False = teknik not.
    '''

    # I3GenieInfo -> her P frame'e (L3 kesiminden ONCE, S frame'ler kesimden
    # etkilenmesin diye)
    if is_genie:
        tray.Add(PropagateGenieInfo, name + "_genie_info")

    if apply_l3_cut:
        # L3 scripti olaylari ATMAZ, sadece bool yazar -> kesimi burada uygula.
        #
        # Tercih sirasi:
        #   1) L3_oscNext_bool  = IC2018_LE_L3_Full AND Data_quality_bool
        #      (data quality dahil: LID errata yok VE SLOP filtresini gecmemis)
        #   2) IC2018_LE_L3_bools.IC2018_LE_L3_Full  (data quality HARIC)
        #
        # SLOP trigger'lari cok genis zaman penceresi (birkac ms) yuzunden
        # DeepCore kriterlerini tesadufen saglayabiliyor ama duzgun simule
        # edilmiyorlar; bu yuzden (1) tercih edilmeli.
        def l3_cut(frame):
            if "L3_oscNext_bool" in frame:
                return bool(frame["L3_oscNext_bool"].value)
            if "IC2018_LE_L3_bools" in frame:
                return bool(frame["IC2018_LE_L3_bools"]["IC2018_LE_L3_Full"])
            return False
        tray.Add(l3_cut, name + "_L3_cut")

    tray.Add(oscNext_L4_common_variables, name + "_common",
             cleaned_pulses=cleaned_pulses,
             uncleaned_pulses=uncleaned_pulses)

    if compute_hit_statistics:
        tray.Add(oscNext_L4_hit_statistics, name + "_hitstats",
                 cleaned_pulses=cleaned_pulses)

    tray.Add(oscNext_L4_noise_cut_variables, name + "_noise_vars",
             fill_ratio_vertex=L4_FIRST_HLC_KEY,
             cleaned_pulses=cleaned_pulses,
             micro_count_pulses=(uncleaned_pulses if micro_count_uncleaned
                                 else None))

    tray.Add(oscNext_L4_atm_muon_classifier_variables, name + "_muon_vars",
             uncleaned_pulses=uncleaned_pulses,
             cleaned_pulses=cleaned_pulses,
             run_optional=run_optional)

    if apply_cut:
        if classifier_model_dir is None:
            raise ValueError("apply_cut=True icin classifier_model_dir gerekli")
        tray.Add(compute_L4_cut, name + "_cut",
                 classifier_model_dir=classifier_model_dir)
