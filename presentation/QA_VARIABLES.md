# How each L4 variable is computed — answers for questions after the talk

This note is for the question "how do you actually compute that one?".  It is
not part of the talk.

Each entry has two parts.  **Say this** is a short spoken answer, in the same
plain English as the script.  **Detail** is what to fall back on if the person
keeps pulling.  Nothing here has to be memorised; the spoken answer alone is
enough for most questions.

Everything below was read out of `oscnext_l4/variables.py`, so it matches the
code that produced the numbers on the slides.

---

## First: the two pulse series

Almost every answer starts here, so it is worth having ready.

**Say this.**  There are two pulse series.  The uncleaned one is
`SplitInIcePulses`, and it covers the whole readout window, about ten
microseconds.  The cleaned one is `SRTTWSplitInIcePulsesDC`.  Level 3 produced
it with SRT cleaning and a time window, and it only holds DeepCore.  Almost
everything at Level 4 starts from the cleaned series.

---

# Noise classifier — the five inputs

## 1. `NchCleaned`

**Say this.**  We do not compute it.  It arrives ready from Level 3.  It is the
number of hit DOMs in the cleaned series.

**Detail.**  Read from `IC2018_LE_L3_Vars.NchCleaned`.  Defined in section
3.5.1 of the reference.  It is also an input to the muon classifier.  That is
why five plus ten inputs come from only fourteen unique variables.

---

## 2. `micro_count`

**Say this.**  It is a four-step chain, and it follows the description in
Table 11 step by step.  We start from the cleaned series.  First a fixed time
window around the trigger.  Then we keep only the DeepCore fiducial DOMs.  Then
we slide a two hundred nanosecond window and keep the densest one.  Then we
count how many DOMs are left.  The answer is a DOM count, not a pulse count.

**Detail.**  The chain, with the parameter values:

| step | module | parameters |
|---|---|---|
| 1 | `I3StaticTWC<I3RecoPulseSeries>` | `WindowMinus=3500`, `WindowPlus=4000` ns, `TriggerConfigIDs=[1010, 1011]`, `TriggerName="I3TriggerHierarchy"` |
| 2 | `I3OMSelection<I3RecoPulseSeries>` | `selectInverse=True`, `OmittedKeys=DeepCoreFiducialDOMs` (the IC86 list) |
| 3 | `I3TimeWindowCleaning<I3RecoPulse>` | `TimeWindow=200` ns |
| 4 | our Python, `_micro_count()` | writes `len(pulse_map)` as an `I3MapStringInt` |

Output key `L4_micro_count`, sub-key `STW_m3500p4000_DTW200`.

Step 2 is not in Table 11, but it is in the original pass2 code, so we kept it.

**If they ask which series it starts from.**  The reference says the cleaned
series.  The original pass2 code starts from the uncleaned one.  We follow the
reference.  We also measured the difference and it is small: the two chains
give the same count in about ninety percent of events.  The two hundred
nanosecond window at the end is what really decides, and noise hits are spread
over ten microseconds either way.

**If they ask about the bug.**  The old chain had an SRT cleaning step in the
middle.  Nothing ever read its output.  The OM selection took the static-window
output instead, so that cleaning step did nothing.
That came from the original pass2 code, not from us.  The original author even
left a comment there, asking whether that series is used at all.

**If they ask how it differs from the Level 3 micro count.**  Level 3 has its
own, `STW9000_DTW300Hits`, with a window of minus four to plus five
microseconds and a three hundred nanosecond slide.  Different parameters, so
the two are not fully correlated, and the classifier takes information from
both.

---

## 3. `iLineFit_speed`

**Say this.**  That one is not our code.  It is the standard IceTray LineFit
segment, run on the cleaned series with no extra parameters.  We read the speed
it fits.  The original pass2 code makes exactly the same call.

**Detail.**

```python
tray.AddSegment(linefit.simple, name + "_iLineFit",
                inputResponse=cleaned_pulses,
                fitName="L4_iLineFit")
```

The classifier reads `L4_iLineFitParams.lf_vel`.  The reference calls the
variable `L4_iLineFit.speed`.  Both names are in our alternatives list, and the
column that is really in the file wins.

---

## 4. `fill_ratio`

**Say this.**  The fill ratio itself comes from the IceTray module, on the
cleaned series.  But the module needs a vertex, and that vertex is ours.  We
give it the position of the first HLC hit.  The original pass2 code does the
same.  The sphere radius parameter is 1.6, also from the original code.

**Detail.**

```python
tray.AddModule("I3FillRatioModule", name + "_FillRatio",
               RecoPulseName       = cleaned_pulses,
               ResultName          = "L4_fill_ratio",
               SphericalRadiusMean = 1.6,
               VertexName          = "L4_first_hlc")
```

The vertex is produced by our own `_first_hlc()`.  It walks the cleaned series and keeps
only HLC pulses, using the `PulseFlags.LC` bit.  It takes the first HLC pulse
per DOM, then picks the earliest one overall.  It writes that time and the DOM
position as an `I3Particle`.  The original used a C++ module called
`FirstHLC<I3RecoPulse>`.  That module is not in every meta-project, so we
wrote the same thing in Python.

The output column is `fillratio_from_mean`, with no underscore after "fill" --
the HDF5 writer names it differently from the reference.

**If they push on the radius.**  This is the honest answer, and it is also next
step two on my last slide.  The reference never says where the vertex is -- the
cross-reference in Table 11 is empty.  And the radius 1.6 carries a comment in
the original code saying it was tuned for GRECO and never re-tuned for oscNext.
The model takes about sixty percent of its gain from this variable.  So that
untuned parameter is the single thing most worth fixing.

---

## 5. `FullTimeLengthRatio`

**Say this.**  Level 3 writes the cleaned and the uncleaned duration
separately, but not their ratio, so we divide them at Level 4.  Cleaned over
uncleaned, so the value sits between zero and one.

**Detail.**  Preferred path: read `CleanedFullTimeLength` and
`UncleanedFullTimeLength` from `IC2018_LE_L3_Vars` and divide.  Fallback, when
the Level 3 map does not have them: measure both durations directly from the
pulse series as `max(time) - min(time)`.

We checked the division against the Level 3 components on 143 events.  Maximum
deviation zero.

**If they ask how we know the direction.**  The text of Table 11 does not say
which duration goes on top.  Figure 13 does: its x axis runs from zero to one.
The other direction would be one or larger and would not fit that axis.

**If they ask why it separates the way it does.**  This is the part I put on
slide 5.  The uncleaned series spans the whole readout window in every event.
So the denominator is nearly constant.  The variable is really the cleaned
duration over ten microseconds.  The median is 0.16 for electron neutrinos and
0.27 for noise, so the noise events have the longer cleaned series.  It
separates well, but the reason is the opposite of what our own docstring used
to claim.

**If they ask about NaN handling.**  We had a bug there.  The guard tested
`uncleaned <= 0`.  That test is False for NaN, so a NaN denominator wrote a
NaN result silently.  Now we test the result with `isfinite` and simply do not
write the variable when it fails.  It then shows up as missing in the HDF5,
That is better than a wrong number.  A ratio above one also raises a warning,
because that would mean the two durations came from different base series.

---

# Muon classifier — the ten inputs

Not trained yet, so these answers are about the code, not about performance.

## From Level 3, no work on our side

`ICVetoHits`, `RTVeto250Hits`, `NchCleaned`, `NAbove200Hits`.

**Say this.**  Four of the ten arrive ready from Level 3.  They are veto hit
counts and the hit count above the DeepCore trigger depth, all defined in
section 3.5.1.

---

## `VICH_nch` — ours

**Say this.**  It counts triggered DOMs in the veto region that could have come
from a muon.  We take the charge-weighted centre of gravity of the hits in the
fiducial volume.  Then, for each veto hit, we work out the speed between that
hit and the centre of gravity.  If the speed falls in the window
the reference gives, the hit counts.

**Detail.**  The speed window is 0.25 to 0.4 metres per nanosecond, and it is
in section 3.4, not in Table 12.  The veto region is the DeepCore Filter's
definition, not the Level 3 fiducial definition.  Those are two different
lists.  We use the uncleaned series, and so does the original pass2 code.  The time
direction is `dt = t_cog - t_hit > 0`.

**One deviation we found and fixed.**  The reference computes the centre of
gravity from hits inside the fiducial volume.  Our code used the whole cleaned
series at first.  In events with a muon, the veto hits pulled the centre of
gravity upward.  In one test the depth moved from minus four hundred metres to
minus forty-three.  The fiducial version is now the default.

**Still open.**  The reference does not say whether the centre of gravity is
charge weighted.  We take it charge weighted.

---

## `accumulated_time` — ours

**Say this.**  It is the time needed to collect seventy-five percent of the
event's charge, in the cleaned series.  The reference states the fraction and
the series, so neither is a guess.

**Detail.**  `fraction=0.75`, cleaned series.  The original pass2 code also
passes the cleaned series to the Dunkman module, so the series choice is
confirmed from two sides.

**Still open.**  The zero point.  Our code measures from the first pulse,
`t[index] - t[0]`.  The reference does not state the reference time, and it
could have been the trigger time instead.  The original implementation is C++
inside a project we do not have.

**Worth adding if the person is interested.**  The reference itself flags this
as one of the few charge-dependent variables in the selection.  pass3 changed
the charge calibration, so this is a variable to watch.

---

## `first_hlc_rho` — ours

**Say this.**  It is the radial distance of the first HLC hit from string 36.
String 36 sits roughly at the centre of DeepCore.  We already build the first HLC
hit for the fill ratio.  This is the same object, with a radius computed from
it.

**Detail.**  `_first_hlc()` produces the particle, `_add_rho_36()` writes the
radius.  The reference's Figure 21 puts this variable highest of the ten in the
muon classifier, so it will matter.

---

## `cog_z`, `z_sigma`, `z_travel` — IceTray

**Say this.**  Those three come from the standard `common_variables` hit
statistics segment, on the cleaned series.  Depth of the centre of gravity, the
RMS of the hit depths, and the total vertical span.  Level 3 deletes the hit
statistics, so we recompute them at Level 4 on the same series.

---

# Questions that cut across all of them

**"Which of these did you write yourself?"**
Five of the fourteen: `micro_count`, `FullTimeLengthRatio`, `VICH_nch`,
`accumulated_time` and `first_hlc_rho`.  Four arrive from Level 3 and five come
from IceTray projects.

**"So the IceTray ones are safe?"**
Mostly, but not entirely, and `fill_ratio` is the example.  The module is
IceTray's, but the vertex and the radius we pass are our choice.  The code is
validated; the parameters are not.

**"How did you validate the five you wrote?"**
Line by line against the reference, and against the original pass2 code.  We
have that code as a file, even though it does not run.  That comparison found four
bugs, all of them silent.  Beyond that, the rates agree with Table 13.  A
direct event-by-event comparison against pass2 Level 4 files is the real test,
and it has not been done yet.

**"Why not just run the original code?"**
The whole body is commented out.  It also needs three projects that are not in
our meta-project: `icecube.oscNext`, `tau_bdt` and `analysis.event_selection`.  We still use the file as a source of parameter
values, and it settled several questions the reference left open.

**"What did the four bugs actually do?"**
An index table was written over the data it indexed.  Event IDs are not unique
between files, so some rows were matched to the wrong event.  A flux event
count was read once and then reused for ten files.  And the noise cleaning step
in the `micro_count` chain was bypassed.  All four produced wrong numbers with
no error raised.

**"Which variable would you fix first?"**
The `fill_ratio` radius.  One parameter, one scan, and it touches sixty percent
of the noise model.
