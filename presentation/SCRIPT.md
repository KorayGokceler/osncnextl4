# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides.  About 2,100 spoken words.  That is roughly 15 minutes at a
normal pace, which leaves time for questions.  The times are cumulative.

The English is kept simple on purpose: short sentences, one idea each.  Every
number is already on a slide or in a plot, so you do not need to memorise
anything.  Point at the plot and say the line.

---

## Slide 1 — Title  ·  0:00–0:45

Thank you.  I have been rebuilding Level 4 of the oscNext selection for pass3.

I want to show you three things today.  First, that my processing gives the
same rates as the reference document.  Second, that the noise classifier now
reaches the performance the reference reports.  Third, what is still not
verified.

One note before I start.  When I say "the reference", I mean the oscNext note,
*Simulations and sample*, version 00.07.  I did not reprocess pass2.  Every
pass2 number you see is quoted from that document.

---

## Slide 2 — The oscNext selection chain  ·  0:45–2:15

This is the selection chain.  These are the reference's own words, from
section 1.3.

Look at the shape of it.  Level 2 is the collaboration filter.  Level 3 uses
simple, fast cuts.  Its job is to make the data and MC agree well enough that
machine learning can be used at all.  Level 4 is that machine learning.  Level 5
removes muons that come in through the corridors in the veto.  Level 6 is the
final reconstruction.  Level 7 cuts on it.

So each level is cheaper than the next one.  That is the design.  And that is
why Level 4 cannot reconstruct anything.  Reconstruction is what Level 6 pays
for.  Level 4 has to separate signal from background using only fast variables.

It has a lot of separating to do.  This table is Table 13.  At Level 3 the muon
rate is about 505 millihertz.  The noise rate is about 37.  The neutrino rate is
about 5.  Background is a hundred times larger than signal.  Level 4 is the level
that turns this around.

*(Only if asked: the sample splits in two at Level 6, a verification sample and
a high-statistics sample.  So there are really two L6 and L7 stages.)*

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  2:15–4:00

Level 4 is two classifiers.  One removes pure noise.  One removes atmospheric
muons.  An event passes if the noise score is at least 0.70 and the muon score
is at least 0.65.  Those two cut values come from the reference, section 3.5.

So why rebuild it?  The original Level 4 code exists, but the whole body is
commented out.  And it needs three projects that are not in our meta-project.
`icecube.oscNext` applies the classifier.  `tau_bdt` gives VICH.
`analysis.event_selection` gives the Dunkman variables.  We have none of them.
So we cannot just switch the old code on.

That means the variables had to be made again.  There are fourteen of them.
Four come ready from Level 3, so there is no work.  Five come from IceTray
projects that we do have.  And five had to be written again, in plain Python,
from the descriptions in the reference.

All the risk is in that last row.  Those five could be wrong.  And they would be
wrong quietly.  A variable computed the wrong way does not raise an error.  It
just gives you a number.

So I compared my code line by line against the reference, and against the
original pass2 code.  We do have that file, even though it does not run.

The comparison found bugs.  Every one of them gave wrong numbers with no error.
That is the kind of bug you only find by looking.

*(Only if asked, and do not list them unprompted: an index table that was
overwriting the data it indexed; event IDs that are not unique between files,
so rows were matched to the wrong events; a flux event count that was read once
and then reused for ten files; and a noise cleaning step in `micro_count` that
was skipped.  That last one comes from the original pass2 code, not from us.
The original author's own comment asks whether that series is used at all.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  4:00–5:30

These are the five inputs of the noise classifier.  The descriptions are Table 11
of the reference, word for word.

I show them as quotes on purpose.  Two of them are not really definitions.

Look at `fill_ratio`.  It says "measure of the geometrical spread of the hits
about some vertex".  Then it says "details here".  That is an empty
cross-reference in the document.  The vertex is never given.  Remember this.  I
come back to it in three slides.

Now look at `FullTimeLengthRatio`.  It tells you it is a ratio of the cleaned and
the uncleaned duration.  It does not tell you which one is on top.

I closed both gaps from outside the document.  The vertex came from the original
pass2 code.  It uses the position of the first HLC hit.  The direction of the
ratio came from Figure 13.  Its axis goes from zero to one.  Only cleaned over
uncleaned can do that.

I want to be open about this.  This is the weakest part of the chain.  I got
these from a code file and from a figure axis, not from a specification.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  5:30–7:00

These are the five inputs, computed on our own pass3 files.  Signal is blue,
noise is red.  This is rate against the variable.

They separate.  That is the first thing you want to see.  But one of them
separates in the opposite direction from what we expected.  I think that is worth
a minute.

Our docstring used to say that `FullTimeLengthRatio` is close to one for a real
event and close to zero for noise.  That is simply wrong.  Measuring it showed
why.

The uncleaned series is `SplitInIcePulses`.  It covers the whole readout window.
That is about ten microseconds, in every event, signal or noise.  So the bottom
of the ratio is almost a constant.  The variable is really just the cleaned
duration divided by ten microseconds.

The median is about 0.16 for electron neutrinos and about 0.27 for noise.  So the
noise events have the longer cleaned series, not the shorter one.  The variable
separates well.  But it works because a low-energy cascade is short in time, not
because noise is long.  It was doing the right thing for the wrong reason.  If we
had trusted the docstring, we would have read this plot backwards.

---

## Slide 6 — Input correlations  ·  7:00–8:00

Before training, there is an obvious question.  Are these five variables saying
the same thing five times?

These are Spearman rank correlations.  Signal on the left, background on the
right.  I compute them separately, because a pair can be correlated in one class
and not in the other.

The answer is no.  They are mostly independent.  Nothing here is a duplicate.
Every input carries something the others do not.

So if one of them turns out to add almost nothing, that means the variable is
weak.  It does not mean another variable already covers it.  Keep that in mind
for the next slide.

---

## Slide 7 — Feature importance  ·  8:00–9:15

This is the gain split of the trained noise model.  For what we do next, this is
the most important plot in the talk.

`fill_ratio` carries about sixty percent of the gain.  `NchCleaned` about
twenty-six percent.  `FullTimeLengthRatio` about eight.  `micro_count` about
four.  And `iLineFit_speed` about one percent.  That last one is almost dead.

Now put this together with slide 4.  The variable the model uses most is the one
whose vertex the reference never defines.  And it has one free parameter, the
spherical radius, set to 1.6.  That value comes from the original code.  The
comment next to it says it was tuned for GRECO and was never re-tuned for
oscNext.

So the model leans hardest on a parameter that was tuned for a different
selection.  Re-tuning it is the best single thing we can do next.  And it is
cheap.  It is one number and a scan.

---

## Slide 8 — How a boosted decision tree works  ·  9:15–10:00

Very short, because you all know this.  One shallow tree is a weak classifier.
Boosting trains many trees in a row.  Each one corrects what the earlier trees
got wrong.

The two engines I compare choose the next tree differently.  AdaBoost gives more
weight to the events it is getting wrong.  Gradient boosting fits the next tree
to the gradient of the loss.  The reference uses LightGBM, section 3.6.1, which
is gradient boosting.

One detail matters later: the output scale.  LightGBM returns a probability,
between zero and one.  So the reference cut of 0.70 works on our model directly.
A pybdt score is not on that scale.  With pybdt you have to find a new threshold
yourself.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  10:00–11:45

We first trained with pybdt.  That is IceCube's own BDT library, and it uses
AdaBoost.  This was a choice against the reference.  The argument was that pybdt
is the in-house tool.

This plot shows six configurations.  Signal efficiency against background
rejection, on the test set.  The table gives the numbers at three working points.

The best one keeps about 94 percent of the signal at 90 percent rejection.  But
at 99 percent rejection it keeps only 66 percent.  The reference target is about
96 percent.  So at the working point that matters, we were thirty points short.

Two remarks.

First, one good property.  Training is deterministic.  Run the same setup twice
and you get exactly the same scores.  So there is nothing to average and no seed
to worry about.

Second, a warning about method.  We used the Kolmogorov–Smirnov p-value to test
for overtraining.  On this data it does not work.  It marked our best model as
overtrained, with p equal to 0.002.  And it passed the worst model, with p equal
to 0.98.  The reason is simple.  We have about a hundred thousand signal events.
With that many events, KS finds differences that are statistically real but
physically meaningless.  We now use something that answers the real question:
the difference between train and test efficiency at the same rejection.

I should also be fair here.  These hyperparameters were chosen by hand.  They
were not tuned as carefully as the reference's.  So the honest statement is not
"AdaBoost is a worse algorithm".  It is "our pybdt setup was costing us a lot".

---

## Slide 10 — Training with LightGBM  ·  11:45–13:15

So we went back to the reference's own method.  LightGBM, with the Table 10
hyperparameters.

Same events.  Same five variables.  Same train and test split.  Same weights.
Scored with the same code.  Only the trainer changed.

At 90 percent rejection, 94 becomes 99.  At 95 percent, 91.7 becomes 98.5.  And
at 99 percent rejection, 65.8 becomes 95.9.

That reaches the reference target.

Two things make me believe it.

First, there is no overtraining.  The gap between train and test efficiency is
one tenth of a point.  And the score distributions sit on top of each other.
That is the next slide.

Second, it reached the target with a handicap.  The reference asks for at least
500 events in a leaf.  That number was tuned on much more noise simulation than
we have.  We only had about 830 background events in the fit.  So that setting
strongly limits what any tree can do.  Training also stopped early, at 122 trees
out of 2,000.  It still met the target.

I want to flag one thing this changed.  We used to think our background
statistics were the bottleneck.  We thought the model could not do better because
there was not enough noise MC.  This measurement showed that was wrong.  The
bottleneck was the engine.

---

## Slide 11 — LightGBM score distribution  ·  13:15–14:00

This is the overtraining check.  The left panel is linear.  The right panel is
the same histogram on a log scale.

You need the log panel.  The tail between the two peaks is where overtraining
would show, and you cannot see it on a linear axis.

Train and test lie on top of each other in both panels.

One thing to read carefully.  The y axis is the training weight, summed per bin.
Each class is normalised to the same total.  So the two areas are equal by
construction.  Please do not read this as "there is as much background as
signal".  For noise the weights happen to be uniform, so the background curves
are really event counts.

And look at where the background stops.  It stops around 0.85.  No background
event in the test set scores higher than that.

---

## Slide 12 — The limit: noise MC statistics  ·  14:00–15:15

If the engine was the bottleneck, what is the limit now?

The test set has about 330,000 signal events and 2,051 noise events.  That is 159
to one.  And the cut removes the noise very fast.  At 90 percent rejection about
100 background events are left above the cut.  At 95 percent, 50.  At 99 percent,
ten.  At 99.5 percent, five.  At 99.9 percent, one.

The reference target sits at 99.2 percent rejection.  Our test set defines that
point with eight events.

So I want to be careful about what I ask for.  What we have is enough to reach
the target.  That is what the last slides showed.  What it is not enough for is to
measure anything above about 99 percent rejection.  The 95 percent row rests on
fifty events, so it is solid.  The 99 percent row rests on ten, so it is at the
edge.

More vuvuzela simulation is useful, and I would like it.  But it is for measuring
the far tail.  It is not blocking the classifier.

---

## Slide 13 — Status and next steps  ·  15:15–16:30

Let me summarise.

Level 3 to Level 4 is rebuilt for pass3, and it runs reliably.  The pass3
production has some half-written files, and one of them kills a whole job.  So
the processing now scans the files first and skips the bad ones.  All fourteen
inputs are found and their column names are fixed.  The rewritten variables were
checked against the reference and against the original code.  And the noise
classifier is trained and reaches the target.

Now what comes next, in order of value.

First, and this is where I would like help.  We should cross-check against the
pass2 Level 4 files.  Those files already contain these variables, computed by
the original code.  Comparing mine against theirs, event by event, is the only
real way to check the five I rewrote.  Everything I said about them today comes
from reading code and reading a figure.  It would also let us train on one pass
and test on the other.  That is a much stronger test than either one alone.  If
anyone knows where those files are, please tell me after the talk.

Second, re-tune the `fill_ratio` radius, for the reason on slide 7.

Third, train the muon classifier.  The CORSIKA background is already processed.
And the train and test split is already done by shower, so copies of the same
air shower cannot end up on both sides.

Fourth, more vuvuzela simulation, to measure the far tail.

Fifth, reprocess the remaining samples.  The muon-neutrino and noise jobs were
stopped early by corrupt input files.

Two limits I should say clearly.  There is no tau neutrino set here, which is a
few percent of the signal.  And the muon background is CORSIKA, not real data, so
there is no data/MC check yet.

Thank you.  I am happy to take questions.

---

# If asked

**"Why not just use the original Level 4 code?"**
The whole body is commented out, and it needs three projects we do not have.  We
keep the file, and it was very useful for parameter values.  But it does not run.

**"How do you know your rewritten variables are correct?"**
I do not know it fully.  That is why it is the first item on the next-steps list.
What I can say is this.  They were compared line by line with the reference and
with the original code.  That comparison found four real bugs.  And the rates
agree with Table 13.  An event-by-event comparison with pass2 L4 files is what
would settle it.

**"Is 95.9 percent meaningful with only ten background events?"**
That is a fair question.  The 99 percent point is defined by ten test-set
background events, so its uncertainty is large.  The 95 percent row has fifty
events and is much more solid.  There LightGBM gives 98.5 against 91.7.  So the
ordering is clear.  The exact value at 99 percent is not.

**"Why is your signal rate higher than Table 13?"**
Because our flux is a simple power law, not a real atmospheric flux with
oscillations.  A factor of about 1.5 is what that should give.  The noise rate
does not depend on that choice, and it agrees to 13 percent.  That agreement is
what tells us the vuvuzela weight unit in pass3 is right.

**"What was the weighting bug?"**
The weights were divided by the number of output HDF5 files instead of the number
of input Level 3 files.  We put ten input files in one output file.  So the
absolute rate was off by a factor of a hundred.  The comparison with Table 13 is
what caught it.  That is the argument for always doing that comparison.

**"Which pulse series does `micro_count` start from?"**
The reference says the cleaned series.  The original pass2 code starts from the
uncleaned one.  We follow the reference.  We also measured the difference, and it
is very small.  The two chains give the same count in about ninety percent of
events.  The 200 nanosecond window is what decides, and noise hits are spread over
ten microseconds either way.

**"Is the 0.70 cut yours or the reference's?"**
The reference's.  It transfers directly because LightGBM returns a probability.
We have not re-optimised it.  And we should not, until the muon classifier
exists, because the Level 4 cut is the two classifiers together.

---

# Speaker's note, not part of the talk

The HDF5 files behind these numbers were made before the `micro_count` cleaning
fix.  The measured effect of that fix is small: the same count in about ninety
percent of events.  But if you reprocess before the talk, check the numbers on
slides 10 and 12 again instead of assuming they stay the same.
