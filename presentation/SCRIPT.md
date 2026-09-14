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

## Slide 9 — Training with pybdt (AdaBoost)  ·  4:51–5:25

This plot shows six pybdt configurations on the test set.  pybdt is IceCube's
own library, and it comes with AdaBoost ready to use.

The x axis is background rejection, the y axis is signal efficiency, and the
black star is the target from the reference.  None of the six curves reaches the
star.

pybdt also gives a Kolmogorov-Smirnov p-value to check for overtraining.  I
tried it, and on this data it does not work.  We compare the train and test
efficiency at the same rejection instead.

---

## Slide 10 — Training with LightGBM  ·  5:25–6:07

This plot shows LightGBM against the best two pybdt models.

I changed one thing only: same events, same variables, same split, same weights,
same scoring code.  Only the trainer is different.  So this is a fair
comparison.

As you can see, LightGBM keeps 95.9 percent of the signal at 99 percent
rejection.  So with LightGBM we reach the target from the reference, and
LightGBM does clearly better than pybdt.

There is no overtraining.  The gap between train and test is very small.

This result changed our minds about something.  We used to believe our background
statistics were the problem.  This measurement showed that the problem was the
engine.

---

## Slide 11 — LightGBM score distribution  ·  6:07–6:32

As you can see from these plots, the classifier separates noise and signal
nicely.  The x axis is the classifier output, from zero to one.  The y axis is
the training weight in each bin.  The left panel is linear and the right one is
on a log scale.

And we also see no overtraining here, because train and test agree.

---

## Slide 12 — The limit: noise MC statistics  ·  6:32–7:30

This plot shows where our noise simulation runs out.

On the left, the x axis is the cut value and the y axis is the percentage of
events that survive.  Blue is the signal we keep and red is the background we
reject.  The dotted line is the reference cut at 0.70.  As you can see, it sits
in a comfortable place: we reject almost all the noise and lose almost no
signal.

On the right, the axes are the ones from slides 9 and 10.  The grey lines mark
where the background statistics run out, at a hundred, ten and one remaining
events.  The star sits past the "ten events" line.

So we have enough noise MC to reach the target, but not enough to measure
anything above 99 percent rejection.  More vuvuzela simulation is useful, but we
need it for the far tail, not for the target.

---

## Slide 13 — Status and next steps  ·  7:30–8:30

The classifier works.  Level 3 to Level 4 is rebuilt for pass3 and runs
reliably, all fourteen inputs are found, and the noise classifier reaches the
target.  The five variables I wrote myself are probably correct, but not proven.

That is the first next step.  The pass2 Level 4 files already contain these
variables, produced by the original code, so we can compare them event by event.
It would also let us train on one pass and test on the other.  So if you know
where those files are, I would be glad to hear it afterwards.

The second step is the `fill_ratio` radius, from slide 7.  Then the muon
classifier, then more vuvuzela simulation, then the rest of the samples.

Two limits.  We have no tau neutrino set, and our muon background is CORSIKA,
not real data, so we cannot compare data and MC yet.

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

**On the Table 10 hyperparameters fitting our sample size.**
They do not, and the model reached the target anyway.  The reference asks for at
least 500 events in one leaf, and somebody tuned that number on far more noise
simulation than we have.  We only had about 830 background events in the fit, so
that setting limits the trees hard.  The training also stopped early, at 122
trees out of 2,000.

**On the y axis of the score distribution.**
It is the training weight, added up in each bin.  We normalise each class to the
same total, so the blue area and the red area are equal by definition.  The plot
does not say that we have as much background as signal.  We do not.  For noise
the weights are all equal, so the red curves are really event counts.

**On 95.9 percent resting on ten background events.**
That is a fair point.  The test set has about 330,000 signal events against
2,051 noise, a ratio of 159 to one, and the cut eats what is left of the noise
fast.  About 100 background events survive at 90 percent rejection, 50 at 95,
ten at 99.  The reference target sits at 99.2 percent, and only eight events
define that point.  So the uncertainty on the 99 percent number is large.  The 95 percent row has fifty events and is
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
