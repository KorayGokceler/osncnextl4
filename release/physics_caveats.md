## Physics caveats

These are the ways this package differs from the official oscNext pass2 L4 (`oscNext_meta V01-00-07`, technical note v00.07). Agreement figures are measured unless stated otherwise.

**Rewritten modules.** This runs on IceTray v1.17.0 (py3-v4.4.2), which lacks three modules the L4 segment calls: `FirstHLC<I3RecoPulse>`, `CalculateVariables` (the Dunkman variables) and `I3CutL7Module` (VICH). They are reimplemented in `oscnext_l4/rewritten.py`. We ran this over pass2 L3 and compared event by event with the real pass2 L4 files (56,301 events: νe, νμ, ντ, MuonGun, noise). 14 of the 15 compared quantities are bitwise identical, including every control quantity (iLineFit, hit statistics, L3 inputs).

**accumulated_time agrees in 99.84 % of events.** This follows `CalculateVariables.cxx`: the time of the last pulse whose cumulative charge lies in (0.5 Q, 0.75 Q], relative to the first pulse. It is 0 when no pulse falls in that band, and it is not written when the cleaned series has 4 or fewer DOMs. The remaining 0.16 % differ by at most 286 ns. The original sorts pulses by time with `std::sort`, which is not stable, so the order of pulses with equal times is unspecified. We use a stable sort. Table 12 of the note ("time to reach 75 %", i.e. the pulse at the crossing) agrees with pass2 in only 0.98 % of events; `--accumulated-time-note` selects that reading.

**micro_count follows the production, not the note.** The chain is `SplitInIcePulses` (uncleaned) → static time window [−3.5, +4] µs around triggers 1010/1011 → DeepCore fiducial DOMs → 200 ns sliding window → number of DOMs. The production also runs a SeededRT step whose output nothing reads; we omit it and the result is bitwise identical to pass2. Table 11 says to start from the cleaned series; `--micro-count-cleaned` does that and changes the value in about 9 % of events.

**first_hlc.** Hits tied in time go to the highest OMKey, as in `FirstHLC.cxx`. An event with no HLC pulse still gets a vertex at (1000, 1000, 1000) m, t = 1e10 ns, which gives `L4_first_hlc_rho` = 1407.3 m. `L4_fill_ratio` is computed about that vertex. Both are bitwise identical to pass2.

**VICH.** The reference hit is the pulse closest in time to the first trigger with config id 1010 or 1011. Pulses are selected with four conditions on distance and time difference, using the uncleaned series pulse by pulse. This agrees 100 % with pass2 simulation. That simulation contains no 1010 triggers, so that path is untested. Events with no matching trigger get 0. `L4_VICH_nch` and `L4_VICH_npulses` are written as `I3Double`; the production writes `I3Int`.

**FullTimeLengthRatio.** pass3 L3 does not store this ratio, so it is always computed at L4 as `CleanedFullTimeLength / UncleanedFullTimeLength` and written to `L4_FullTimeLengthRatio`. The model reads it from there, not from `IC2018_LE_L3_Vars`. It is identical to the stored pass2 value over 8,144 events.

**Pulse series and key names.** The cleaned series is `SRTTWSplitInIcePulsesDC` for pass3 and `SRTTWOfflinePulsesDC` for pass2 (set with `--cleaned-pulses`). The uncleaned series is `SplitInIcePulses` in both. Hit statistics and multiplicity are always written as `SRTTWSplitInIcePulsesDCHitStatistics` / `…HitMultiplicity`. On pass2 this pass3-style name holds values computed from `SRTTWOfflinePulsesDC`, which match pass2's own `SRTTWOfflinePulsesDCHitStatistics` bitwise.

**The L4 boolean.** `L4_oscNext_bool` = `L4_NoiseClassifier_ProbNu >= 0.70` AND `L4_MuonClassifier_Data_ProbNu >= 0.65`, the production thresholds. It is written for every event; nothing is filtered at L4. Frames failing `L3_oscNext_bool` and non-`InIceSplit` frames are dropped, as in production. If `L3_oscNext_bool` is absent, `IC2018_LE_L3_Full` is used instead, which on pass3 does not include the data-quality cut. A missing input variable is passed as NaN and takes LightGBM's learned default branch, as in the production.

**These are our models, not the collaboration's.** They use the same algorithm and inputs, the Table 10 hyperparameters, and class-balanced weights rescaled to [0, 1]. They differ in a learning rate of 0.05 with early stopping on part of the training set, and a 50 % training fraction instead of the production's 33 % (noise) and 2 % (neutrinos). The thresholds are the production's, but applied to different models they do not select the same events. {{AT_DEFAULT_CUT}} L5's tightened cuts (0.85 and 0.90) were tuned on the production scores.

**Signal sample and weights.** {{SIGNAL}}

**Muon background.** {{MUON_BACKGROUND}} The production used pass2 detector data from 18 runs (three per year, 2012-2017), after the noise cut at 0.70. The key `L4_MuonClassifier_Data_ProbNu` is written whatever background was used.

**Frame keys not produced.** `L4_QR_Box` (slc-veto is absent), `L4_ToI*` and `L4_separation_in_cogs` (not BDT inputs; available with `--run-optional`), `L4_Dunkman_*_Variables` (only its two extracted fields are written), `L4_SRTTWPulses` and `L4_MuonClassifier_MuonGun_ProbNu`.

**Limitations inherited from the production.** fill_ratio uses `SphericalRadiusMean = 1.6`, tuned for GRECO and never re-tuned for oscNext; it carries about 61 % of the noise model's gain. Models learn step functions in `first_hlc_rho` at the string positions, so tiny GCD differences can move events across a learned boundary.
