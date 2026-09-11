'''
HDF5 booking without hdfwriter.

If the meta-project was built without HDF5 support, icecube.hdfwriter does not
exist.  This module does the same job in pure Python + pytables and reproduces
exactly the table layout the notebook expects:

    /IC2018_LE_L3_Vars      Run, Event, SubEvent, NchCleaned, ICVetoHits, ...
    /L4_VICH_nch            Run, Event, SubEvent, value
    /L4_iLineFitParams      Run, Event, SubEvent, LFVel, LFVelX, ...
    /SRTTWOfflinePulsesDCHitStatistics
                            Run, Event, SubEvent, cog_x, cog_y, cog_z, ...

This is not a general tabulation -- it extracts scalar fields only.  That is
exactly what L4 variable production needs (pulse series are not booked).

Usage:

    from oscnext_l4.booker import SimpleBooker
    tray.Add(SimpleBooker, "booker", Output="out.hdf5", Keys=[...])
'''

import numpy as np

import os, sys
from .env import require_icetray
require_icetray()
from icecube import icetray, dataclasses


# ---------------------------------------------------------------------------
# Extracting scalar fields from a frame object
# ---------------------------------------------------------------------------

# Field lists for the known types.  An introspection fallback exists too, but
# an explicit list is both faster and more predictable.
KNOWN_FIELDS = {
    "I3Particle": ["time", "energy", "length", "speed",
                   "zenith", "azimuth", "x", "y", "z"],
    "I3LineFitParams": ["LFVel", "LFVelX", "LFVelY", "LFVelZ", "NHits"],
    "I3HitStatisticsValues": ["cog_x", "cog_y", "cog_z",
                              "min_pulse_time", "max_pulse_time",
                              "q_max_doms", "q_tot_pulses",
                              "z_min", "z_max", "z_mean", "z_sigma", "z_travel"],
    "I3HitMultiplicityValues": ["n_hit_strings", "n_hit_doms",
                                "n_hit_doms_one_pulse", "n_pulses"],
    "I3TensorOfInertiaFitParams": ["mineval", "evalratio", "eval2", "eval3"],
    "I3EventHeader": ["run_id", "sub_run_id", "event_id", "sub_event_id"],
}

# The I3EventHeader time fields are needed for the livetime calculation.
# start_time / end_time are I3Time objects -> expanded as MJD day + seconds.
_TIME_FIELDS = {"start_time": "time_start_mjd", "end_time": "time_end_mjd"}

# I3Position / I3Direction fields are expanded into flat columns
_VECTOR_EXPAND = {
    "pos": ["x", "y", "z"],
    "dir": ["zenith", "azimuth"],
    "cog": ["x", "y", "z"],
}


def _num(v):
    '''Can this be converted to a number?  Covers bools and enums too.'''
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    # enum (fit_status, ParticleType, ...)
    if hasattr(v, "real") and not isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


def extract_scalars(obj):
    '''
    Extract a {column_name: float} dict from a frame object.

    Tried in order:
      1) map-like (has keys())            -> one column per key
      2) has a .value field               -> a single "value" column
      3) known type                       -> the KNOWN_FIELDS list
      4) introspection                    -> numeric public fields
    '''
    tname = type(obj).__name__
    out = {}

    # 1) map benzeri: I3MapStringDouble, I3MapStringInt, I3MCWeightDict
    if hasattr(obj, "keys") and not hasattr(obj, "pos"):
        try:
            for k in obj.keys():
                v = _num(obj[k])
                if v is not None:
                    out[str(k)] = v
            if out:
                return out
        except Exception:
            pass

    # 2) I3Double, I3Int, I3Bool
    if hasattr(obj, "value") and not hasattr(obj, "pos"):
        v = _num(obj.value)
        if v is not None:
            return {"value": v}

    # 3a) I3EventHeader: zaman alanlarini MJD olarak ac
    if tname == "I3EventHeader":
        for attr, prefix in _TIME_FIELDS.items():
            try:
                t = getattr(obj, attr)
                out[prefix + "_day"] = float(t.mod_julian_day)
                out[prefix + "_sec"] = float(t.mod_julian_sec)
                out[prefix + "_ns"]  = float(t.mod_julian_nano_sec)
            except Exception:
                pass

    # 3) bilinen tip
    fields = KNOWN_FIELDS.get(tname)
    if fields:
        for f in fields:
            try:
                v = _num(getattr(obj, f))
            except Exception:
                v = None
            if v is not None:
                out[f] = v
        # For I3Particle, pos/dir are already there as x,y,z,zenith,azimuth
        if out:
            return out

    # 4) introspection
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            v = getattr(obj, name)
        except Exception:
            continue
        if callable(v):
            continue
        # I3Position / I3Direction gibi bilesik alanlari ac
        sub = _VECTOR_EXPAND.get(name)
        if sub:
            for s in sub:
                try:
                    sv = _num(getattr(v, s))
                except Exception:
                    sv = None
                if sv is not None:
                    out[f"{name}_{s}"] = sv
            continue
        nv = _num(v)
        if nv is not None:
            out[name] = nv
    return out


# ---------------------------------------------------------------------------
# Tray modulu
# ---------------------------------------------------------------------------

class SimpleBooker(icetray.I3ConditionalModule):
    '''hdfwriter'a ihtiyac duymadan HDF5 yazan booking modulu.'''

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Output", "Cikti .hdf5 dosyasi", "output.hdf5")
        self.AddParameter("Keys", "Book edilecek frame objeleri", [])
        self.AddParameter("SubEventStreams", "Islenecek sub-event stream'ler",
                          ["InIceSplit"])
        self.AddParameter("Verbose", "Eksik anahtarlari raporla", True)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.output = self.GetParameter("Output")
        self.keys = list(self.GetParameter("Keys"))
        self.streams = list(self.GetParameter("SubEventStreams"))
        self.verbose = self.GetParameter("Verbose")

        # table name -> {column: [values]}
        self.tables = {}
        # table name -> rows seen so far (for index alignment)
        self.n_rows = {}
        self.n_frames = 0
        self.missing = {}

    def Physics(self, frame):
        if "I3EventHeader" not in frame:
            self.PushFrame(frame)
            return
        hdr = frame["I3EventHeader"]
        if self.streams and hdr.sub_event_stream not in self.streams:
            self.PushFrame(frame)
            return

        idx = (int(hdr.run_id), int(hdr.event_id), int(hdr.sub_event_id))
        self.n_frames += 1

        for key in self.keys:
            if key not in frame:
                self.missing[key] = self.missing.get(key, 0) + 1
                continue
            obj = frame[key]
            # mask/union ise atla -- pulse serilerini book etmiyoruz
            if hasattr(obj, "apply"):
                continue
            try:
                vals = extract_scalars(obj)
            except Exception as e:
                if self.verbose:
                    icetray.logging.log_warn(
                        "SimpleBooker: %s cikarilamadi: %s" % (key, e))
                continue
            if not vals:
                continue

            t = self.tables.setdefault(key, {"Run": [], "Event": [], "SubEvent": []})
            n = self.n_rows.get(key, 0)
            t["Run"].append(idx[0]); t["Event"].append(idx[1]); t["SubEvent"].append(idx[2])
            for col, v in vals.items():
                if col in ("Run", "Event", "SubEvent"):
                    col = col + "_"          # cakismayi onle
                lst = t.setdefault(col, [np.nan] * n)
                # onceki satirlarda bu kolon yoksa NaN ile doldur
                while len(lst) < n:
                    lst.append(np.nan)
                lst.append(v)
            self.n_rows[key] = n + 1
            # bu satirda gorunmeyen kolonlari NaN ile hizala
            for col, lst in t.items():
                while len(lst) < self.n_rows[key]:
                    lst.append(np.nan)

        self.PushFrame(frame)

    def Finish(self):
        import tables

        if not self.tables:
            icetray.logging.log_warn(
                "SimpleBooker: no events were booked at all! "
                "Check the SubEventStreams parameter.")

        filters = tables.Filters(complevel=5, complib="zlib")
        with tables.open_file(self.output, "w", filters=filters) as h5:
            for name, cols in self.tables.items():
                n = self.n_rows[name]
                arrays = {}
                for col, lst in cols.items():
                    while len(lst) < n:
                        lst.append(np.nan)
                    a = np.asarray(lst)
                    if col in ("Run", "Event", "SubEvent"):
                        a = a.astype(np.int64)
                    else:
                        a = a.astype(np.float64)
                    arrays[col] = a
                order = ["Run", "Event", "SubEvent"] + \
                        sorted(c for c in arrays if c not in ("Run", "Event", "SubEvent"))
                dt = np.dtype([(c, arrays[c].dtype) for c in order])
                rec = np.empty(n, dtype=dt)
                for c in order:
                    rec[c] = arrays[c]
                # tablo adlarinda '/' olamaz
                safe = name.replace("/", "_")
                h5.create_table("/", safe, rec, title=name)

        print(f"SimpleBooker: {self.n_frames} frame -> {len(self.tables)} tablo "
              f"-> {self.output}")
        if self.verbose and self.missing:
            print("  Hic bulunamayan / eksik anahtarlar:")
            for k, c in sorted(self.missing.items(), key=lambda kv: -kv[1]):
                frac = 100.0 * c / max(self.n_frames, 1)
                flag = "  <-- NEVER PRESENT" if frac > 99.9 else ""
                print(f"    {k:48s} {c:7d} frame ({frac:5.1f}%){flag}")


def add_booker(tray, name, output, keys, sub_event_streams=("InIceSplit",)):
    '''
    Use hdfwriter when it is available, otherwise SimpleBooker.
    '''
    try:
        from icecube import hdfwriter
        tray.Add(hdfwriter.I3HDFWriter, name,
                 Output=output, Keys=keys,
                 SubEventStreams=list(sub_event_streams))
        print("Booking: icecube.hdfwriter")
        return "hdfwriter"
    except ImportError:
        tray.Add(SimpleBooker, name,
                 Output=output, Keys=keys,
                 SubEventStreams=list(sub_event_streams))
        print("Booking: SimpleBooker (no hdfwriter, pytables fallback)")
        return "simple"
