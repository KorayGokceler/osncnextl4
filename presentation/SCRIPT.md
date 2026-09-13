# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides.  Spoken text is roughly 2,000 words, about 15 minutes at a
normal pace, leaving room for questions.  Timings are cumulative and assume you
do not read the tables aloud — point at them instead.

Every number below is on a slide or in a plot.  Nothing here needs to be
memorised; the script says where to look.

---

## Slide 1 — Title  ·  0:00–0:45

Thanks.  I have been rebuilding Level 4 of the oscNext selection for pass3,
and today I want to show you three things: that the processing reproduces the
rates in the reference document, that the noise classifier now reaches the
performance the reference reports, and what is still unverified.

One framing note before I start.  Everything I call "the reference" is the
oscNext note, *Simulations and sample*, version 00.07.  I did not reprocess
pass2 for any of this — wherever a pass2 number appears, it is quoted from that
document.

---

## Slide 2 — The oscNext selection chain  ·  0:45–2:15

This is the selection chain as the reference lists it, section 1.3, quoted
directly.

The part I want you to notice is the shape of it.  Level 2 is the
collaboration-wide filter.  Level 3 is simple, fast cuts whose job is to get
the data–MC agreement good enough that machine learning can be applied at all.
Level 4 is that machine learning.  Level 5 removes muons that sneak through the
veto along the corridors, Level 6 is the final reconstruction, and Level 7 cuts
on it.

So each level is cheaper than the one that follows.  That is the whole design,
and it is why Level 4 is not allowed to reconstruct anything — reconstruction is
what Level 6 pays for.  Level 4 has to separate signal from background using
only quantities you can compute quickly.

And it has a lot of separating to do.  At Level 3 — this is Table 13 — the
atmospheric muon rate is about 505 millihertz and the noise rate about 37,
against roughly 5 millihertz of neutrinos.  Background outnumbers signal by two
orders of magnitude.  Level 4 is the level that inverts that.

*(If the sample-split question comes up: the sample splits in two at Level 6,
a verification sample and a high-statistics one, so there are strictly two
distinct L6 and L7 stages.)*

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  2:15–4:00

Level 4 is two independent classifiers — one against pure noise, one against
atmospheric muons — applied as a conjunction.  An event survives if the noise
classifier gives it at least 0.70 and the muon classifier at least 0.65.  Those
thresholds are the reference's, section 3.5.

Now, why rebuild it at all.  The original Level 4 tray segment exists, but its
entire body is commented out, and it depends on three projects that are not in
the meta-project we run: `icecube.oscNext`, which provides the classifier
application module; `tau_bdt`, which provides VICH; and
`analysis.event_selection`, which provides the Dunkman variables.  None of them
are available, so the segment cannot simply be switched on.

That meant re-deriving the discriminating variables.  Fourteen go into the two
classifiers.  Four come ready from Level 3, so there is nothing to do.  Five
come from IceTray projects we do have.  And five had to be rewritten in pure
Python from their definitions in the reference.

All of the risk sits in that last row.  Those five are the ones that could be
quietly wrong — and "quietly" is the operative word, because a variable computed
incorrectly does not raise an error, it just produces a number.  So the rewrite
was compared line by line against the reference and against the original pass2
segment, which we do have as source even though it does not run.

That comparison found bugs.  All of them produced wrong numbers with no error
raised — which is exactly the class of bug you only find by looking.

*(Only if asked, and do not enumerate unprompted: an index table that was
overwriting the data it indexed, event IDs that are not unique across files so
rows were matched to the wrong events, a flux event count that was read once and
frozen across a ten-file chunk, and a noise-cleaning step in `micro_count` that
was bypassed.  That last one is inherited from the original pass2 code, not
introduced by us — the original author's own comment questions whether that
series is used.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  4:00–5:30

These are the five noise-classifier inputs, with the reference's Table 11
descriptions quoted word for word.  I am showing them quoted deliberately,
because two of them are not actually definitions.

Look at `fill_ratio`.  The description reads "measure of the geometrical spread
of the hits about some vertex", and then, in the document, an unfilled
cross-reference: "details here".  The vertex is never specified.  That matters
more than it looks, and I will come back to it.

And `FullTimeLengthRatio` — the description tells you it is a ratio of the
cleaned and uncleaned durations, but not which way round.

Both gaps were closed from outside the document.  The vertex came from the
original pass2 code, which passes the position of the first HLC hit.  The
direction of the ratio came from Figure 13: the axis runs from zero to one, and
only cleaned-over-uncleaned can do that.

I want to be honest that this is the weakest part of the chain.  We recovered
these from a code file and a figure axis, not from a specification.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  5:30–7:00

These are the five inputs computed on our own pass3 processing — signal in blue,
noise in red, rate against variable.

They separate, which is the first thing you want to see.  But one of them
separates in the opposite direction to what we expected, and I think it is worth
a minute.

The docstring in our code used to say that `FullTimeLengthRatio` is near one for
a real event and near zero for noise.  That is simply wrong, and measuring it
showed why.  The uncleaned series is `SplitInIcePulses`, which spans the entire
readout window — about ten microseconds — in every single event, signal or
noise.  So the denominator is effectively a constant, and the variable is just
the cleaned duration divided by ten microseconds.

The median is about 0.16 for electron-neutrino events and 0.27 for noise.  The
noise events have the *longer* cleaned series.  It separates well — but because
a low-energy cascade is compact in time, not because noise is spread out.  The
variable was doing the right thing for the wrong reason, and if we had trusted
the docstring we would have drawn the wrong conclusion from it.

---

## Slide 6 — Input correlations  ·  7:00–8:00

Before training, the obvious question: are these five variables saying the same
thing five times?

These are Spearman rank correlations, signal on the left, background on the
right, computed separately because a pair can be correlated in one class and not
the other.

The answer is no, they are largely independent.  Nothing here is redundant —
every input is carrying information the others are not.  Which means if one of
them turns out to contribute very little, that is a statement about the variable
being weak, not about it being duplicated.

Hold that thought for the next slide.

---

## Slide 7 — Feature importance  ·  8:00–9:15

This is the gain split of the trained noise model, and it is the most
consequential plot in the talk for what we do next.

`fill_ratio` carries about sixty percent of the gain.  `NchCleaned` about
twenty-six.  `FullTimeLengthRatio` eight, `micro_count` four, and
`iLineFit_speed` about one percent — effectively dead.

Put that together with the previous slide and with slide 4.  The variable the
model leans on hardest is the one whose vertex the reference does not define,
and whose only free parameter — the spherical radius, set to 1.6 — comes
straight from the original code with a comment attached saying it was optimised
for GRECO and has not been re-optimised for oscNext.

So we are relying, more than on anything else, on a parameter that was tuned for
a different event selection.  Re-tuning it is the single highest-value knob we
have, and it is cheap: it is one number and a scan.

---

## Slide 8 — How a boosted decision tree works  ·  9:15–10:00

Very briefly, because you all know this.  A shallow decision tree is a weak
classifier; boosting trains many of them in sequence, each correcting what the
previous ones got wrong.

The two engines I will compare differ in how the next tree is chosen.  AdaBoost
re-weights the events the ensemble is getting wrong.  Gradient boosting fits the
next tree to the gradient of the loss.  The reference uses LightGBM, section
3.6.1, which is gradient boosting.

The one thing that matters downstream is the output scale.  LightGBM returns a
probability, between zero and one, so the reference's cut value of 0.70
transfers to our model directly.  A pybdt score does not live on that scale, so
with pybdt the threshold has to be recalibrated from scratch.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  10:00–11:45

We first trained with pybdt, which is IceCube's own BDT library and implements
AdaBoost.  That was a deliberate deviation from the reference — the argument was
that it is the in-house tool.

This is six configurations, signal efficiency against background rejection on
the test set, and the table gives the numbers at three working points.

The best configuration keeps about 94 percent of the signal at 90 percent
rejection, but only 66 percent at 99 percent rejection.  The reference target is
around 96.  So at the working point that actually matters, we are missing it by
thirty points.

Two remarks.  First, a genuinely good property: training is deterministic — run
the same configuration twice and you get bit-identical scores, so there is
nothing to average over and no seed to worry about.

Second, a warning about methodology.  We used the Kolmogorov–Smirnov p-value on
the train and test score distributions as an overtraining test, and on this data
it is useless.  It flagged our best model as overtrained, p equals 0.002, and
cleared the worst one at 0.98.  The reason is that with of order a hundred
thousand signal events, KS detects differences that are statistically
significant and physically irrelevant.  We replaced it with something that
answers the question we actually care about: the difference between train and
test efficiency at the same rejection.

I should be fair about the framing here.  These hyperparameters were hand-picked;
they were not tuned to the same effort as the reference's.  So the honest
statement is not "AdaBoost is a worse algorithm" — it is "our pybdt setup was
costing us a lot".

---

## Slide 10 — Training with LightGBM  ·  11:45–13:15

So we went back to the reference's own method: LightGBM, with the Table 10
hyperparameters.

Same events, same five variables, same train/test split, same weights, scored
with the same code.  Only the trainer changed.

At 90 percent rejection, 94 goes to 99.  At 95 percent, 91.7 goes to 98.5.  And
at 99 percent rejection — the one that matters — 65.8 goes to 95.9.

That meets the reference target.

Two things make me believe it.  First, there is no overtraining: the train–test
efficiency gap is a tenth of a point, and the score distributions sit on top of
each other, which is the next slide.

Second, it got there handicapped.  The reference's minimum-leaf-size setting is
500 events, and it was tuned on far more noise simulation than we have — we only
had about 830 background events in the fit, so that setting is a hard constraint
on what any tree can carve out.  Training also stopped early, at 122 trees out
of a budget of 2,000.  It met the target anyway.

I want to flag one thing that this measurement changed.  We had previously
concluded that our background statistics were the bottleneck — that the model
could not do better because there was not enough noise MC.  This retracted that.
The bottleneck was the engine.

---

## Slide 11 — LightGBM score distribution  ·  13:15–14:00

This is the overtraining check.  Left panel linear, right panel the identical
histogram on a log scale — you need the log panel, because the tail between the
two peaks is where overtraining would show and it is invisible on a linear axis.

Train and test lie on top of each other in both.

One thing to read carefully: the y axis is the training weight summed per bin,
and each class is normalised to the same total.  So the two areas are equal by
construction — do not read this as "there is as much background as signal".
For noise the weights happen to be uniform, so the background curves are
effectively event counts.

And notice where the background stops — around 0.85.  No background event in the
test set scores higher than that.

---

## Slide 12 — The limit: noise MC statistics  ·  14:00–15:15

So if the engine was the bottleneck, what is the limit now?

The test set has about 330,000 signal events against 2,051 noise — a ratio of
159 to one.  And the cut eats what is left of the noise very fast.  At 90 percent
rejection about 100 background events remain above the cut.  At 95 percent, 50.
At 99 percent, ten.  At 99.5, five.  At 99.9, one.

The reference's target working point is 99.2 percent rejection.  Our test set
defines that point with eight events.

So I want to be precise about the ask, because it is easy to overstate.  What we
have is enough to *reach* the target — that is what the previous slides showed.
What it is not enough for is to *measure* anything to the right of about 99
percent rejection.  The 95 percent row, resting on fifty events, is firm.  The
99 percent row, resting on ten, is at the edge of what is measurable.

More vuvuzela simulation is genuinely valuable for pinning down the far tail.
It is not blocking the classifier.

---

## Slide 13 — Status and next steps  ·  15:15–16:30

To summarise where this is.

Level 3 to Level 4 is rebuilt for pass3 and runs robustly — the pass3 production
contains half-written files that kill a tray outright, so the processing scans
and blacklists them.  All fourteen classifier inputs are located and their column
names pinned down.  The rewritten variables have been validated against the
reference and the original segment.  And the noise classifier is trained and
reaches the target.

What is next, in order of value.

First, and this is the one I would most like help with: cross-checking against
the pass2 Level 4 files.  Those files already contain these variables, computed
by the original code.  Comparing ours against theirs event by event is the only
way to actually validate the five we rewrote — everything I have shown you about
them rests on a code reading and a figure axis.  It would also let us train on
one pass and test on the other, which is a much stronger statement than either
alone.  If anyone knows where those files live, please tell me afterwards.

Second, re-tune the `fill_ratio` radius, for the reason on slide 7.

Third, train the muon classifier — the CORSIKA background is processed and the
train/test split is already done at shower level, so copies of the same air
shower cannot leak across the split.

Fourth, more vuvuzela simulation, to measure the far tail rather than to reach
the target.

And fifth, reprocess the remaining samples — the muon-neutrino and noise sets
were cut short by corrupt input files.

Two limitations to state plainly.  There is no tau-neutrino set in this, which is
a few percent of the signal.  And the muon background is CORSIKA rather than real
data, so there is no data/MC check yet.

Thank you — happy to take questions.

---

# If asked

**"Why not just use the original Level 4 code?"**
Its body is commented out in full, and it needs three projects that are not in
the meta-project.  We have the file as a source of parameter values — and it was
genuinely useful for that — but it does not run.

**"How do you know your rewritten variables are right?"**
I do not, fully, and that is next step one.  What I can say is that they were
compared line by line against the reference and against the original code, that
the comparison found four real bugs, and that the rates come out consistent with
Table 13.  A direct event-by-event comparison against pass2 L4 files is what
would actually settle it.

**"Is 95.9 percent significant, given ten background events?"**
Fair challenge.  The 99 percent working point is defined by ten test-set
background events, so the uncertainty on that number is large.  The 95 percent
row, with fifty events, is much firmer, and there LightGBM gives 98.5 against
AdaBoost's 91.7.  The ordering is not in doubt; the exact value at 99 percent is.

**"Why is your signal rate high compared with Table 13?"**
Because the flux is an invented power law, not a real atmospheric flux with
oscillations.  A factor of about 1.5 is what that should give.  The noise rate,
which does not depend on that choice, agrees to 13 percent — and that agreement
is what validates the vuvuzela weight unit in pass3.

**"What was the weighting bug?"**
The weights were divided by the number of HDF5 output files instead of the
number of Level 3 input files.  With ten input files per output chunk that is a
factor of a hundred in the absolute rate.  The rate comparison against Table 13
is what caught it, which is the argument for always doing that comparison.

**"Which pulse series does `micro_count` start from?"**
The reference says the cleaned series; the original pass2 code starts from the
uncleaned one.  We follow the reference.  We also measured the difference and it
is almost nil — the two chains give an identical count in about ninety percent
of events, because the closing 200-nanosecond window is what decides and noise
hits are spread over ten microseconds either way.

**"Is the cut value 0.70 yours or the reference's?"**
The reference's.  It transfers directly because LightGBM outputs a probability.
We have not re-optimised it, and we should not until the muon classifier exists,
since the Level 4 cut is the conjunction of the two.

---

# Speaker's caveat, not for the talk

The HDF5 files behind these numbers were produced before the `micro_count`
cleaning fix landed.  The measured effect of that fix is small — identical
counts in about ninety percent of events — but if you reprocess before the talk,
re-check the numbers on slides 10 and 12 rather than assuming they carry over.
