# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides, about 2,300 spoken words.  At 150 words a minute, which is a
normal rehearsed pace, that is 15 minutes.  Read slowly and unrehearsed it runs
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

## Slide 3 — What L4 does, and what had to be rebuilt  ·  1:19–2:24

Level 4 uses two classifiers.  One removes noise, one removes muons.  An event
passes with a noise score of 0.70 or higher **and** a muon score of 0.65 or
higher.  Both values come from the reference.

The original Level 4 code still exists, but somebody commented out the whole
body.  It also needs three projects that we do not have.  So I built the
variables again.

The table on the right splits them into three groups.  Fourteen in total.  Four
come ready from Level 3.  Five come from IceTray projects.  And I wrote five
myself, from the descriptions in the reference.

The red row is the point of this slide.  A variable computed the wrong way does
not raise an error.  It just gives you a number.  So I compared my code against
the reference line by line, and against the original pass2 code.  That found
four bugs, all of them silent.

That comparison raised a second problem, and it takes us to the next slide.

*(Only if someone asks.  An index table was writing over the data.  Event IDs
are not unique between files, so some rows went to the wrong event.  A flux
event count was read once and then reused for ten files.  And one noise
cleaning step in `micro_count` was skipped.  That last bug comes from the
original pass2 code, not from us.  The original author even left a comment
there, asking whether that series is used at all.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  2:24–3:32

A line-by-line comparison needs a definition to compare against.  For two of
these five variables, the reference does not give one.

This table has the five noise inputs, with the descriptions from Table 11, word
for word.  I show their exact wording so the gaps are visible.

The fourth row is `fill_ratio`.  It describes the spread of the hits about some
vertex, and then it says "details here", in red.  That is an empty
cross-reference.  So the reference never tells us which vertex to use.  This row
comes back on slide 7.

The last row is `FullTimeLengthRatio`.  It is a ratio of the cleaned and the
uncleaned duration.  The document does not say which one goes on top.

I found both answers outside the document.  The vertex came from the original
pass2 code, which uses the first HLC hit.  The direction came from Figure 13,
because its axis runs from zero to one.

This is the weakest part of the chain.  I took these answers from a code file and
from a figure axis.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  3:32–4:56

This plot shows the five noise inputs on our own pass3 files.

Each panel is one variable.  The x axis is the variable, the y axis is rate in
hertz per bin on a log scale.  Blue is neutrino and red is noise.

The top left panel is `NchCleaned`.  The red curve dies quickly toward the
right.  The blue curve runs out to a hundred and forty.  So small events are
noise and large events are neutrinos, as we expect.  In the top right panel the
two curves sit almost on top of each other.  That one comes back on slide 7.

The bottom right panel surprised us.  Our docstring used to describe
`FullTimeLengthRatio` as close to one for signal and close to zero for noise.
That is simply wrong.

The uncleaned series covers the whole readout window in every event, about ten
microseconds.  So the bottom of the ratio is almost a constant.  The variable is
really the cleaned duration divided by ten microseconds.  The median is 0.16 for
electron neutrinos and 0.27 for noise.  The noise events have the **longer**
cleaned series.

The variable still separates well, but for the opposite reason.  A low-energy
cascade is short in time.  Noise is not long.  We had the right variable and the
wrong story.

---

## Slide 6 — Input correlations  ·  4:56–5:45

This plot shows how much the five inputs overlap with each other.

Both axes list the five inputs.  The colour is the Spearman rank correlation:
dark red is plus one, dark blue is minus one, white is zero.  Signal is on the
left and background on the right.

The diagonal is only each variable against itself.  Away from it, almost
everything is pale.  The strongest pair on the signal side is `NchCleaned` with
`micro_count`, at 0.71, and both of them count hit DOMs.  On the background side
that pair drops to 0.10.

So these are five different variables.  That matters for the next slide.  If one
of them adds almost nothing, the variable is weak.  It is not weak because
another variable already covers it.

---

## Slide 7 — Feature importance  ·  5:45–7:03

This plot shows how much each input contributed to the trained model.

The y axis lists the five inputs, sorted.  The x axis is gain in percent, which
means how much each variable improved the splits during training.

`fill_ratio` takes about sixty percent.  `NchCleaned` takes twenty-six.  The last
three take eight, four and one.

`iLineFit_speed`, at the bottom, is almost dead.  After the last slide we know
how to read that.  It is not hidden inside another variable.  It simply does not
separate, and slide 5 already showed it.

The top bar is the important one.  The model leans on `fill_ratio` more than on
everything else together.  And `fill_ratio` is the variable from slide 4, with
the empty cross-reference.

There is a second layer to that.  `fill_ratio` has one free parameter, the
spherical radius, set to 1.6 in the original code.  The comment next to that
value says somebody tuned it for GRECO.  Nobody tuned it again for oscNext.

So our model depends most on a parameter from a different event selection.  That
is a risk, and it is also the cheapest improvement we have: one parameter and one
scan.

That closes the variables.  Now the classifier.

---

## Slide 8 — How a boosted decision tree works  ·  7:03–7:41

This will be short.  Boosting trains many shallow trees, one after another, and
each new tree corrects the mistakes of the earlier ones.  AdaBoost gives more
weight to the events it gets wrong.  Gradient boosting fits the next tree to the
gradient of the loss.  The reference uses LightGBM, which is gradient boosting.

The detail I need is the output scale.  LightGBM gives a probability between zero
and one, so the reference cut of 0.70 works on our model directly.  A pybdt score
is not on that scale, so with pybdt the reference threshold means nothing.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  7:41–9:06

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

## Slide 10 — Training with LightGBM  ·  9:06–10:31

This plot shows LightGBM against the best two pybdt models.

I changed one thing only: same events, same variables, same split, same weights,
same scoring code.  Only the trainer is different, which makes this a fair
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

## Slide 11 — LightGBM score distribution  ·  10:31–11:46

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

## Slide 12 — The limit: noise MC statistics  ·  11:46–13:14

This plot shows where our noise simulation runs out, from two sides.  The engine
was the problem and we fixed it, so this is the limit now.

The left panel is about safety.  The x axis is the cut value.  The y axis is the
percentage of events that survive.  Blue is the signal we keep and red is the
background we reject.  The dotted line is the reference cut at 0.70.

The red curve is already near a hundred by 0.2.  The blue curve stays flat until
about 0.85.  The reference cut sits inside that window, which is comfortable.

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

## Slide 13 — Status and next steps  ·  13:14–14:41

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

Two limits, clearly.  We have no tau neutrino set, which is a few percent of the
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

**On whether the rewritten variables are correct.**
I do not know it completely, and that is the first item on my list.  But I
compared them with the reference line by line, I compared them with the original
code and found four real bugs, and the rates agree with Table 13.

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
agrees to 13 percent, which tells us the vuvuzela weight unit in pass3 is
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
