# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides, about 2,300 spoken words.  At 150 words a minute, a normal
rehearsed pace, that is 15 minutes.  Read slowly and unrehearsed it runs
to 16 and a half, so one run-through with a timer is worth it.  The times below
add up from the start.

The script is cut to fit 15 minutes, so it says one thing per point and moves
on.  Anything that had to go is in the follow-up section at the end, ready if
somebody asks for it.

Four rules I followed while writing this.

**One thread.**  Level 4 must separate signal from background with cheap
variables only.  I wrote five of those variables myself, and they may be wrong.
That risk is stated on slide 3, carried through slides 4 to 7, and paid off on
slide 13.

**The same pattern for every plot.**  What the plot shows, then the axes and the
colours, then what can be seen in it, then what it means.  Always in that order,
and always opening with the same words.

**No questions.**  Every sentence states something.

**No commands.**  The script describes the slide instead of telling the audience
where to look.

The English is simple on purpose.  Short sentences, one idea each.

---

## Slide 1 — Title  ·  0:00–0:30

I have been working on Level 4 of the oscNext selection for a while now.  I am
rebuilding it for pass3.

Level 4 removes noise and atmospheric muons.  I had to write part of it again
from scratch.

One note on sources before I start.  Jana gave me a document about the oscNext
selection.  It is written for pass2.  When I say "the reference" in this talk, I
mean that document.  I did not process pass2 myself.

---

## Slide 2 — The oscNext selection chain  ·  0:30–1:19

This table shows the selection chain.

Level 2 is the collaboration-wide filter.  For us that is the DeepCore Filter.
Level 3 uses simple, fast cuts.  Its job is to make the data and the MC agree.
They must agree before machine learning works.  Level 4 is that machine
learning.  Level 5 removes muons from the corridors.  Level 6 reconstructs, and
Level 7 cuts on the reconstruction.

Each level is cheaper than the next one.  So **Level 4 may not reconstruct
anything.**  Reconstruction is expensive, and Level 6 pays for it.  Level 4 gets
fast variables only.

The small table shows the size of the job.  At Level 3 the muon rate is about
505 millihertz, noise is about 37, and neutrinos are about 5.  Background is a
hundred times larger than signal, and Level 4 turns that around.

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  1:19–1:59

Level 4 uses two classifiers.  One removes noise, one removes muons.  An event
passes with a noise score of 0.70 or higher **and** a muon score of 0.65 or
higher.  Both values come from the reference.

The original Level 4 code still exists.  But it needs three projects that we do
not have.  So I built the variables again.

The table on the right splits them into three groups.  Fourteen variables in
total.  Four come ready from Level 3.  Five come from IceTray projects that we
do have.  And I wrote five of them myself, in plain Python, from the
descriptions in the reference.

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  1:59–3:04

These are the five inputs of the noise classifier.  The descriptions come from
Table 11, word for word.

`NchCleaned` is the number of hit DOMs in the cleaned series.  That one arrives
ready from Level 3, so we do not compute it.

`micro_count` is the largest number of DOMs inside a two hundred nanosecond
window.  We build it in four steps.  A fixed window around the trigger time,
then only the DeepCore fiducial DOMs, then the two hundred nanosecond slide,
then a count.

`iLineFit_speed` is the speed from a straight line fit to the hits.  That is the
standard IceTray segment, run on the cleaned series.

`fill_ratio` is how full a sphere around the event is.  The IceTray module does
the work, and we give it the centre of the sphere.  The centre is the position
of the first HLC hit.

`FullTimeLengthRatio` is the cleaned duration over the uncleaned duration.
Level 3 writes both of them, but not the ratio, so we divide them at Level 4.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  3:04–3:32

This plot shows the five noise inputs on our own pass3 files.

Each panel is one variable.  The x axis is the variable, the y axis is rate in
hertz per bin on a log scale.  Blue is neutrino and red is noise.

From these plots you can already guess how much each variable separates.
Slide 7 puts numbers on that.

The reference shows the same five distributions, in Figures 12 and 13.

---

## Slide 6 — Input correlations  ·  3:32–3:59

This plot shows how strongly the five inputs are related to each other.

The strongest pair on the signal side is `NchCleaned` with `micro_count`, at
0.71, and both of them count hit DOMs.  On the background side that pair drops
to 0.10.

So the five inputs do not repeat each other.

That matters on the next slide.  If one of them scores low there, the variable
is simply weak.

---

## Slide 7 — Feature importance  ·  3:59–4:30

This plot shows how much each input contributed to the trained model.

`fill_ratio` takes about sixty percent.  `NchCleaned` takes twenty-six.  The
last three take eight, four and one.

So the model uses `fill_ratio` more than the other four together.  That
variable has one free parameter, the spherical radius.  It is set to 1.6, and
that value was optimised for GRECO, not for oscNext.  Tuning it for oscNext
could help a lot.

That closes the variables.  Now the classifier.

---

## Slide 8 — How a boosted decision tree works  ·  4:30–4:51

This will be short.  Boosting trains many shallow trees, one after another, and
each new tree corrects the mistakes of the earlier ones.  AdaBoost gives more
weight to the events it gets wrong.  Gradient boosting fits the next tree to the
gradient of the loss.  The reference uses LightGBM.  That is gradient boosting.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  4:51–6:16

This plot shows six pybdt configurations on the test set.  Slides 10 and 12 use
the same axes.

The x axis is background rejection in percent, from 80 to 100, and good is to the
right.  The y axis is signal efficiency, and good is up.  So the best place is
the top right corner.  Each coloured curve is one configuration, and the black
star is the target from the reference.

All the curves sit high on the left, near 95.  They hold that level, and then
after about 97 percent they fall off a cliff.  The star sits above every curve,
in empty space.

That gap is the story of this slide.  The best configuration keeps 94 percent of
the signal at 90 percent rejection, but only 66 percent at 99.  The target is
about 96.  So we were thirty points short at the point that matters.

One warning about method.  We used the Kolmogorov–Smirnov p-value to test for
overtraining.  On this data it does not work.  It called our best model
overtrained and passed the worst one.  We now compare the train and test
efficiency at the same rejection instead.

And I should be fair.  We chose these hyperparameters by hand, so the honest
statement is not that AdaBoost is a worse algorithm.  Our pybdt setup was costing
us a lot.

---

## Slide 10 — Training with LightGBM  ·  6:16–7:40

This plot shows LightGBM against the best two pybdt models.

I changed one thing only: same events, same variables, same split, same weights,
same scoring code.  Only the trainer is different.  So this is a fair
comparison.

The axes are the ones from the last slide.  The green solid curve is LightGBM and
the two dashed curves are pybdt.  The dashed curves start to fall at about 95
percent.  The green curve stays flat, almost at a hundred, all the way to 99.
And the green curve passes through the star.

The table carries the numbers.  At 90 percent rejection, 94 becomes 99.  At 95
percent, 91.7 becomes 98.5.  At 99 percent, 65.8 becomes 95.9.  That reaches the
target of the reference.

Two things make me believe it.  There is no overtraining.  Train and test differ
by one tenth of a point, and the next slide shows that.

The model also reached the target with a handicap.  The reference asks for at
least 500 events in one leaf.  Somebody tuned that number on far more noise
simulation than we have.  We only had about 830 background events in the fit.
The training also stopped early, at 122 trees out of 2,000.

This result changed our minds about something.  We used to believe our background
statistics were the problem.  This measurement showed that the problem was the
engine.

---

## Slide 11 — LightGBM score distribution  ·  7:40–8:56

This plot shows the classifier output for train and test.  It exists for one job:
to show that the model is not overtrained.

The x axis is the classifier output from zero to one.  The y axis is the training
weight per bin.  There are four curves: signal and background, train and test.
Both panels hold the same histogram, linear on the left and log on the right.

In the left panel the signal piles up against one and the background against
zero.  The middle is almost empty.  That is a working classifier.

Overtraining would show in the tail between the two peaks.  On a linear axis that
tail lies flat on the floor.  On the log axis it becomes visible.  And there the
train and test curves follow each other bin by bin.  So the check passes.

One caution on the y axis.  We normalise each class to the same total, so the two
areas are equal by definition.  This slide does not say that we have as much
background as signal.

The red curves stop around 0.85.  No background event in the test set scores
higher than that.

---

## Slide 12 — The limit: noise MC statistics  ·  8:56–10:25

This plot shows where our noise simulation runs out, from two sides.  The engine
was the problem and we fixed it, so this is the limit now.

The left panel is about safety.  The x axis is the cut value.  The y axis is the
percentage of events that survive.  Blue is the signal we keep and red is the
background we reject.  The dotted line is the reference cut at 0.70.

The red curve is already near a hundred by 0.2.  The blue curve stays flat until
about 0.85.  The reference cut sits inside that window.  That is a comfortable place.

The right panel is about the limit.  The axes are the ones from slides 9 and 10.
The grey vertical lines mark where the background statistics run out, at a
hundred, ten and one remaining events.  The star sits past the "ten events" line.

Our test set has about 330,000 signal events and 2,051 noise events.  At 90
percent rejection about 100 background events survive.  At 95 percent, 50.  At 99
percent, ten.  The reference target sits at 99.2 percent, and only eight events
define that point.

So I want to be careful about my request.  We have enough noise MC to reach the
target.  We do not have enough to measure anything above 99 percent rejection.
More vuvuzela simulation is useful, but we need it for the far tail, not for the
target.

---

## Slide 13 — Status and next steps  ·  10:25–11:52

Two things to take away from this.

The classifier works.  Level 3 to Level 4 is rebuilt for pass3 and runs
reliably.  All fourteen inputs are found.  And the noise classifier reaches the
target.

The five rewritten variables are probably correct, but not proven.  That is the
first item here.

We should compare our variables with the pass2 Level 4 files.  Those files
already contain these variables, produced by the original code.  So we can
compare them event by event.  This is the only real way to check the five I wrote
myself.  It would also let us train on one pass and test on the other.  So if you
know where those files are, I would be glad to hear it afterwards.

The second step is the `fill_ratio` radius, from slide 7.  One parameter, one
scan, and it touches sixty percent of the model.

Then the muon classifier, with the CORSIKA background that is already processed.
Then more vuvuzela simulation for the far tail.  And then the rest of the
samples, because corrupt input files stopped two jobs early.

Two limits, clearly.  We have no tau neutrino set.  That is a few percent of the
signal.  And our muon background is CORSIKA, not real data, so we cannot compare
data and MC yet.

Thank you.  I am happy to take questions.

---

# Likely follow-ups

Some of this was in the talk before it was cut to 15 minutes.  It is all ready to
say if somebody asks.

**On the four bugs.**
An index table was writing over the data.  Event IDs are not unique between
files, so some rows went to the wrong event.  A flux event count was read once
and then reused for ten files.  And one noise cleaning step in `micro_count` was
skipped.  That last one comes from the original pass2 code, not from us.

**On the three missing projects.**
`icecube.oscNext` applies the classifier.  `tau_bdt` gives VICH.
`analysis.event_selection` gives the Dunkman variables.  Our meta-project has
none of them.

**On reusing the original Level 4 code.**
Somebody commented out the whole body.  We keep the file, and it helped a lot
with parameter values, but it does not run.

**On the fill_ratio vertex.**
The reference does not give it.  Table 11 describes the spread of the hits about
some vertex, and then says "details here".  That cross-reference is empty.  We
took the vertex from the original pass2 code, and it uses the position of the
first HLC hit.

**On whether the rewritten variables are correct.**
I do not know it completely, and that is the first item on my list.  But I
compared them with the reference line by line, I compared them with the original
code and found four real bugs, and the rates agree with Table 13.

**On FullTimeLengthRatio separating the opposite way.**
Our own docstring used to describe it as close to one for signal and close to
zero for noise.  That is wrong.  The uncleaned series covers the whole readout
window in every event, about ten microseconds.  So the denominator is nearly
constant, and the variable is really the cleaned duration over ten
microseconds.  The median is 0.16 for electron neutrinos and 0.27 for noise, so
the noise events have the longer cleaned series.  It separates well, but for the
opposite reason: a low-energy cascade is short in time.

**On 95.9 percent resting on ten background events.**
That is a fair point.  Ten test-set background events define the 99 percent
number, so its uncertainty is large.  The 95 percent row has fifty events and is
much firmer, and there LightGBM gives 98.5 against 91.7.  The order is clear, the
exact value at 99 percent is not.

**On pybdt being deterministic.**
It is, and that is a real advantage.  The same setup run twice gives exactly the
same scores, so we never have to average and never have to worry about a seed.

**On the x axis of slide 12 running past one.**
The cut scan adds two points outside the real range so the curve closes at both
ends.  Only the region between zero and one is a real cut.  The straight fall on
the far right is a line between two points, not a measurement.

**On our signal rate sitting above Table 13.**
Our flux is a simple power law, not a real atmospheric flux with oscillations.  A
factor of about 1.5 is normal.  The noise rate does not depend on that choice and
agrees to 13 percent.  That tells us the vuvuzela weight unit in pass3 is
correct.

**On the weighting bug.**
We divided the weights by the number of output HDF5 files instead of the number
of input Level 3 files.  Ten input files go into one output file, so the rate was
wrong by a factor of a hundred.  The comparison with Table 13 found it.

**On the pulse series behind `micro_count`.**
The reference says the cleaned series and the original pass2 code starts from the
uncleaned one.  We follow the reference.  We measured the difference and it is
very small: the same count in about ninety percent of events.

**On the 0.70 cut.**
It comes from the reference, and we can use it directly because LightGBM gives a
probability.  We should not change it yet, because the Level 4 cut uses both
classifiers together and the muon one does not exist.

**On the split at Level 6.**
The sample splits in two there, a verification sample and a high-statistics
sample.  So there are really two L6 stages and two L7 stages.

---

# Note for the speaker, not part of the talk

We made the HDF5 files for these numbers before the `micro_count` cleaning fix.
The effect of that fix is small, with the same count in about ninety percent of
events.  But a new processing run before the talk means the numbers on slides 10
and 12 need a re-check.
