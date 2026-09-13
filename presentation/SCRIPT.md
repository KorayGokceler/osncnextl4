# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides.  About 3,300 spoken words, so around 24 minutes at a normal
speed.  The times add up from the start.

With only 15 minutes, three cuts work and nothing breaks.  Slide 8 can go
completely, with its output-scale sentence moved into slide 9.  The first half
of slide 9 can go, leaving the table and the gap to the star.  The left-panel
description on slide 12 can go, leaving the right panel.  Those three save about
eight minutes.  Slide 4, slide 7 and the first item on slide 13 carry the
thread, so they stay.

Four rules I followed while writing this.

**One thread.**  Level 4 must separate signal from background with cheap
variables only.  I wrote five of those variables myself.  So the talk has two
open points: my variables may be wrong, and my classifier may not be good
enough.  Each slide moves one of them forward, and the links between slides are
in the text.

**The same pattern for every plot.**  What the plot shows, then the axes and the
colours, then what can be seen in it, then what it means.  Always in that order.
The audience learns the pattern once and then follows it without effort.

**No questions.**  Every sentence states something.  There is never a pause
where the room is supposed to answer.

**No commands.**  The script never tells the audience to look, remember or
notice anything.  It describes the slide, and their eyes follow by themselves.

The English is simple on purpose.  Short sentences, one idea each.

---

## Slide 1 — Title  ·  0:00–0:50

Thank you.  I have been rebuilding Level 4 of the oscNext selection for pass3.

Level 4 is the step that removes noise and atmospheric muons.  I had to write
part of it again from scratch.  So this talk has two open points.  The first one
is my five rewritten variables.  The second one is the classifier itself.

The short answer to both comes now.  The classifier is good enough.  The
variables are probably correct, but I cannot prove it yet.  The rest of the talk
is the evidence.

One note on sources.  "The reference" means the oscNext note, *Simulations and
sample*, version 00.07.  I did not process pass2 myself.  Every pass2 number in
this talk comes from that document.

---

## Slide 2 — The oscNext selection chain  ·  0:50–2:30

I will start with the place of Level 4 in the chain, because that explains its
constraints.

This table is the chain, in the reference's own words, from section 1.3.  The
middle column gives the job of each level.  Level 2 is the collaboration filter.
Level 3 uses simple, fast cuts, and its job is to make the data and the MC
agree.  They must agree before machine learning is possible at all.  Level 4 is
that machine learning.  Level 5 removes muons that sneak in through the
corridors.  Level 6 does the full reconstruction.  Level 7 cuts on the
reconstruction.

The order matters.  Each level is cheaper than the next one.  That is the whole
design, and it gives Level 4 a hard rule.  **Level 4 may not reconstruct
anything.**  Reconstruction is expensive, and Level 6 pays for it.  Level 4 gets
fast variables only.

That rule runs through the rest of the talk.  It is the reason this work is
about fourteen simple numbers instead of about a fit.

The small table at the bottom gives the size of the job.  At Level 3 the muon
rate is about 505 millihertz.  Noise is about 37.  Neutrinos are about 5.  So
background is a hundred times larger than signal.  Level 4 turns that around,
with cheap variables only.

*(Only if someone asks: the sample splits in two at Level 6.  One part is a
verification sample, the other has higher statistics.  So there are really two
L6 stages and two L7 stages.)*

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  2:30–4:30

Level 4 does the job with two classifiers.  One removes pure noise.  One removes
atmospheric muons.  An event passes if the noise score is 0.70 or higher **and**
the muon score is 0.65 or higher.  Both cut values come from the reference,
section 3.5.

Now the problem.  The original Level 4 code still exists, but somebody commented
out the whole body.  It also needs three projects, and our meta-project has none
of them.  `icecube.oscNext` applies the classifier.  `tau_bdt` gives VICH.
`analysis.event_selection` gives the Dunkman variables.  So we cannot simply
switch the old code on.  I had to build the variables again.

The table on the right splits them into three groups.  There are fourteen
variables in total.  Four come ready from Level 3, so there is no work there.
Five come from IceTray projects that we do have.  And I wrote five of them
myself, in plain Python, from the descriptions in the reference.

The red row is the point of this slide.  All the risk sits in those five.  And
there is a reason I keep coming back to them.  A variable computed the wrong way
does not raise an error.  It gives you a number.  The job runs, the plots look
fine, and the classifier trains happily on wrong data.

So I compared my code with the reference, line by line.  I also compared it with
the original pass2 code, because we still have that file even though it does not
run.  The comparison found bugs.  Every one of them gave wrong numbers, and none
of them gave an error.  Bugs like this only turn up if someone goes looking.

That comparison also raised a second problem, and it takes us to the next slide.

*(Only if someone asks.  An index table was writing over the data.  Event IDs
are not unique between files, so some rows went to the wrong event.  A flux event
count was read once and then reused for ten files.  And one noise cleaning step
in `micro_count` was skipped.  That last bug comes from the original pass2 code,
not from us.  The original author even left a comment there about that series.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  4:30–6:00

Here is the second problem.  A line-by-line comparison needs a definition to
compare against.  For two of these five variables, the reference does not give
one.

This table has the five noise inputs.  The descriptions come from Table 11, word
for word.  I show their exact wording on purpose, so the gaps are visible.

The fourth row is `fill_ratio`.  It says "measure of the geometrical spread of
the hits about some vertex".  Then it says "details here", in red.  That is an
empty cross-reference in the document.  Nobody filled it in.  So the reference
never tells us which vertex to use.  This row comes back on slide 7, and it
becomes the most important thing in the talk.

The last row is `FullTimeLengthRatio`.  It tells us this is a ratio of the
cleaned and the uncleaned duration.  It does not tell us which one goes on top.

I found both answers outside the document.  For the vertex, I read the original
pass2 code, and it uses the position of the first HLC hit.  For the direction of
the ratio, I used Figure 13.  Its axis runs from zero to one, and only cleaned
over uncleaned can do that.

I want to be open about this.  This is the weakest part of the chain.  I took
these answers from a code file and from a figure axis.  Nobody wrote them down
properly.

So the variables are built, and two of them rest on guesses.  The next slide
shows what they actually look like.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  6:00–8:00

This plot shows the five noise inputs, computed on our own pass3 files.

Each panel is one variable.  The x axis is the variable itself.  The y axis is
rate, in hertz per bin, on a log scale.  Blue is neutrino and red is noise.

In the top left panel, `NchCleaned`, the red curve dies quickly toward the right,
while the blue curve keeps going out to a hundred and forty.  The top middle
panel, `micro_count`, behaves the same way in the same direction.  In the top
right panel, `iLineFit_speed`, the two curves sit almost on top of each other.
That one comes back on slide 7.  In the bottom left panel, `fill_ratio`, both
curves are noisy, but the red one is pushed toward zero and the blue one is
flatter.

So the variables separate, and mostly in the direction we expect.  Small events
are noise and large events are neutrinos.

The bottom right panel needs a minute of its own.  It is
`FullTimeLengthRatio`, and it surprised us.

Our own docstring used to describe this variable.  It said the value is close to
one for a real event, and close to zero for noise.  That is simply wrong.  We
measured it, and now we understand why.

The uncleaned series is `SplitInIcePulses`.  It covers the whole readout window,
about ten microseconds, in every event, signal or noise.  So the bottom of the
ratio is almost a constant.  The variable is really the cleaned duration,
divided by ten microseconds.

In that panel the blue curve peaks lower than the red one.  The median is about
0.16 for electron neutrinos and about 0.27 for noise.  So the noise events have
the **longer** cleaned series, not the shorter one.

The variable still separates well, but the reason is the opposite of what we
wrote down.  A low-energy cascade is short in time.  Noise is not long.  We had
the right variable and the wrong story.  With the old docstring in hand, we would
have read this plot backwards.

---

## Slide 6 — Input correlations  ·  8:00–9:15

This plot shows how much the five inputs overlap with each other.

Both axes list the five inputs in the same order.  The colour is the Spearman
rank correlation.  Dark red is plus one, dark blue is minus one, and white is
zero.  Each box also carries the number.  Signal is on the left and background
is on the right.  They are separate, because a pair can be correlated in one
class and not in the other.

The diagonal is dark red in both panels, and that is only each variable against
itself.  Away from the diagonal, almost everything is pale.  On the signal side
the strongest pair is `NchCleaned` with `micro_count`, at 0.71.  Both of them
count hit DOMs, so that one makes sense.  Everything else is small.  On the
background side even that pair drops to 0.10.

So these are five different variables.  None of them is a copy of another one.
That matters for the next slide.  If one of these variables adds almost nothing
to the model, the variable is weak.  It is not weak because another variable
already covers it.

---

## Slide 7 — Feature importance  ·  9:15–11:00

This plot shows how much each input contributed to the trained noise model.

The y axis lists the five inputs, sorted.  The x axis is gain, in percent.  Gain
means how much each variable improved the splits during training.  Each bar
carries its number at the end.

`fill_ratio` takes about sixty percent.  `NchCleaned` takes twenty-six.  Then
`FullTimeLengthRatio` at eight, `micro_count` at four, and `iLineFit_speed` at
one percent.

The bottom bar comes first.  `iLineFit_speed` is almost dead, and after the last
slide we know how to read that.  It is not hidden inside another variable.  It
simply does not separate.  Slide 5 already showed it: its blue and red curves sat
on top of each other.

The top bar is the important one.  The model leans on `fill_ratio` more than on
everything else together.  And `fill_ratio` is the variable from slide 4, with
the empty cross-reference.  The reference never tells us where its vertex is.

There is a second layer to that.  `fill_ratio` has one free parameter, the
spherical radius.  Its value is 1.6, and I took it from the original code.  The
comment next to that value says somebody tuned it for GRECO.  Nobody tuned it
again for oscNext.

So our model depends most on a parameter from a different event selection.  That
is a risk.  It is also the cheapest improvement available to us: one parameter
and one scan.  It is the second item on my last slide.

That closes the variables.  Now the classifier.

---

## Slide 8 — How a boosted decision tree works  ·  11:00–11:50

This will be short, because you all know it.  I need it for one detail only.

One shallow tree is a weak classifier.  Boosting trains many trees, one after
another.  Each new tree corrects the mistakes of the earlier ones.

The two engines pick the next tree in different ways.  AdaBoost gives more weight
to the events it gets wrong.  Gradient boosting fits the next tree to the
gradient of the loss.  The reference uses LightGBM, section 3.6.1, and LightGBM
is gradient boosting.

The detail I need is the output scale.  LightGBM gives you a probability, between
zero and one.  So the cut value 0.70 from the reference works on our model
directly.  A pybdt score is not on that scale.  With pybdt, the reference
threshold means nothing, and you have to find your own.

That is part of the price on the next slide.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  11:50–13:45

We trained with pybdt first.  That is IceCube's own BDT library, and it uses
AdaBoost.  We chose it instead of the reference method, and the argument was
simple.  pybdt is our in-house tool.

This plot shows six pybdt configurations on the test set.  Slides 10 and 12 use
the same axes, so I will set them up carefully here.

The x axis is background rejection, in percent, from 80 to 100.  Good is to the
right.  The y axis is signal efficiency, also in percent.  Good is up.  So the
best place on this plot is the top right corner.  Each coloured curve is one
configuration, and the black star is the target from the reference.

On the left, at 80 percent rejection, all the curves sit high, near 95.  They
hold that level for a while.  Then, after about 97 percent, they fall off a
cliff.  And the star sits above every single curve, in empty space.

That gap is the whole story of this slide.  The table carries the numbers.  The
best configuration keeps 94 percent of the signal at 90 percent rejection.  At 99
percent rejection it keeps only 66.  The target is about 96.  So we were thirty
points short at the point that matters.

Two remarks before I move on.

First, one good property.  The training is deterministic.  The same setup run
twice gives exactly the same scores.  So we never have to average, and we never
have to worry about a seed.

Second, a warning about method.  We used the Kolmogorov–Smirnov p-value to test
for overtraining, and on this data it does not work.  It called our best model
overtrained, with p equal to 0.002.  It passed the worst model, with p equal to
0.98.  The reason is simple.  We have about a hundred thousand signal events, and
with so many events KS finds differences that are real in statistics and
meaningless in physics.  We replaced it with a simpler test.  We take the
efficiency on the train set and on the test set, at the same rejection, and we
look at the difference.

And I should be fair about the framing.  We chose these hyperparameters by hand.
We did not tune them as carefully as the reference did.  So I will not tell you
that AdaBoost is a worse algorithm.  I will tell you that our pybdt setup was
costing us a lot.

That was enough reason to go back to the reference.

---

## Slide 10 — Training with LightGBM  ·  13:45–15:30

So we did exactly that.  LightGBM, with the hyperparameters from Table 10.

This plot shows LightGBM against the best two pybdt models.  I changed one thing
only: same events, same five variables, same train and test split, same weights,
same scoring code.  Only the trainer is different, which makes this a fair
comparison.

The axes are the ones from the last slide.  Rejection across, efficiency up, and
the target star at the top right.  The green solid curve is LightGBM.  The two
dashed curves are the pybdt models.

The dashed curves start to fall at about 95 percent.  The green curve stays flat,
almost at a hundred, all the way to 99, and it turns down only at the very end.
The green curve passes through the star.

The table carries the numbers.  At 90 percent rejection, 94 becomes 99.  At 95
percent, 91.7 becomes 98.5.  And at 99 percent rejection, 65.8 becomes 95.9.
That reaches the target of the reference.

Two things make me believe this result.

First, there is no overtraining.  The train and test efficiencies differ by one
tenth of a point.  The next slide shows that directly.

Second, the model reached the target with a handicap.  The reference asks for at
least 500 events in one leaf.  Somebody tuned that number on much more noise
simulation than we have.  We only had about 830 background events in the fit, so
that setting limits the trees very hard.  The training also stopped early, at 122
trees out of 2,000.  The model still reached the target.

And this result changed our minds about something.  Before this, we believed our
background statistics were the problem.  We believed the model could not do
better, because we did not have enough noise MC.  This measurement showed that we
were wrong.  The problem was the engine, not the statistics.

That was the claim.  Here is the proof.

---

## Slide 11 — LightGBM score distribution  ·  15:30–16:30

This plot shows the classifier output for train and test, and it exists for one
job only: to show that the model is not overtrained.

The x axis is the classifier output, from zero to one.  The y axis is the
training weight, added up in each bin.  There are four curves: signal train,
signal test, background train, background test.  Both panels hold the same
histogram.  The left one is linear and the right one is on a log scale.

In the left panel the signal piles up against one, on the right.  The background
piles up against zero, on the left.  The middle is nearly empty.  That is a
working classifier.

The left panel cannot do this slide's job, and that is why the right panel is
there.  Overtraining shows up in the tail between the two peaks.  On a linear
axis that tail lies flat on the floor.  On a log axis it becomes visible.  And in
that middle region of the right panel, the train curve and the test curve follow
each other bin by bin.  They wander together and they never separate.

So the check passes.  Two more details before I move on.

The y axis needs care.  We normalise each class to the same total, so the blue
area and the red area are equal by definition.  This slide does not say that we
have as much background as signal.  We do not.  For noise the weights are all
equal, so the red curves are really event counts.

And the red curves stop around 0.85 on the right panel.  No background event in
the test set scores higher than that.  That number comes back on the next slide.

---

## Slide 12 — The limit: noise MC statistics  ·  16:30–17:45

This plot shows where our noise simulation runs out, from two sides.  The
engine was the problem and we fixed it, so this is the limit now.

The left panel is about safety.  The x axis is the cut value on the classifier
output.  The y axis is the percentage of events that survive.  Blue is the signal
we keep and red is the background we reject.  The dotted vertical line is the
reference cut, at 0.70.

The red curve climbs almost vertically from zero, and by 0.2 it is already near a
hundred.  The blue curve stays flat at a hundred until about 0.85.  So between
those two values we throw away almost all the noise and lose almost no signal,
and the reference cut sits inside that window.  We are in a comfortable place.

The right panel is about the limit.  The axes are the ones from slides 9 and 10:
rejection across, efficiency up, target star.  The grey vertical lines are the
new information.  They mark where the background statistics run out, at a
hundred, ten and one remaining background events.  The star sits past the "ten
events" line.

The numbers behind that are simple.  Our test set has about 330,000 signal events
and 2,051 noise events, so 159 to one.  At 90 percent rejection about 100
background events survive the cut.  At 95 percent, 50.  At 99 percent, ten.  At
99.9 percent, one.  The reference target sits at 99.2 percent rejection, and only
eight events define that point.

So I want to be careful about my request.  We have enough noise MC to reach the
target.  Slide 10 showed that.  We do not have enough to measure anything above
99 percent rejection.  Fifty events support the 95 percent row, so that row is
solid.  Ten events support the 99 percent row, so that row is at the edge.

More vuvuzela simulation is useful and I would like to have it.  But we need it
to measure the far tail, not to reach the target.  It does not block the
classifier.

---

## Slide 13 — Status and next steps  ·  17:45–19:00

I will pull the two open points back together.

On the classifier, the answer is yes.  Level 3 to Level 4 is rebuilt for pass3
and it runs reliably.  The pass3 production contains some half-written files, and
one bad file can kill a whole job, so the code scans the files first and skips
them.  We found all fourteen inputs and fixed their column names.  And the noise
classifier reaches the target of the reference.

On the variables, the answer is "probably, but not proven".  That is the first
item here.

We should compare our variables with the pass2 Level 4 files.  Those files
already contain these variables, and the original code produced them.  So we can
compare mine with theirs, event by event.  This is the only real way to check the
five variables I wrote myself.  Everything I told you about them today came from
reading a code file and reading a figure axis.  This test would replace that with
a measurement.  It would also let us train on one pass and test on the other one.
That is a much stronger test.  So if you know where those files are, I would be
glad to hear it after the talk.

The second step is the `fill_ratio` radius, from slide 7.  One parameter, one
scan, and it touches sixty percent of the model.

The third step is the muon classifier.  The CORSIKA background is ready.  And we
already split train and test by shower, so copies of the same air shower cannot
land on both sides.

The fourth step is more vuvuzela simulation, to measure the far tail.

The fifth step is to process the rest of the samples.  Corrupt input files
stopped the muon-neutrino and noise jobs early.

Two limits, and I want to say them clearly.  We have no tau neutrino set here,
and that is a few percent of the signal.  And our muon background is CORSIKA, not
real data, so we cannot compare data and MC yet.

Thank you.  I am happy to take questions.

---

# Likely follow-ups

These are the eight things people usually pick up on, with an answer ready for
each.

**On reusing the original Level 4 code.**
Somebody commented out the whole body, and it needs three projects that we do not
have.  We keep the file, and it helped us a lot with parameter values.  But it
does not run.

**On whether the rewritten variables are correct.**
I do not know it completely, and that is the first item on my list.  But I can
say three things.  I compared them with the reference, line by line.  I compared
them with the original code, and that found four real bugs.  And the rates agree
with Table 13.  A direct comparison with pass2 L4 files would give us the real
answer.

**On 95.9 percent resting on ten background events.**
That is a fair point.  Ten test-set background events define the 99 percent
number, so its uncertainty is large.  The 95 percent row has fifty events, so it
is much firmer.  There LightGBM gives 98.5 and AdaBoost gives 91.7.  So the order
is clear.  The exact value at 99 percent is not.

**On the x axis of slide 12 running past one.**
Good catch.  The cut scan adds two points outside the real range, so the curve
closes at both ends.  Only the region between zero and one is a real cut.  The
straight fall on the far right is a line between two points, not a measurement.

**On our signal rate sitting above Table 13.**
Our flux is a simple power law, not a real atmospheric flux with oscillations.  A
factor of about 1.5 is normal for that.  The noise rate does not depend on this
choice, and it agrees to 13 percent.  That tells us the vuvuzela weight unit in
pass3 is correct.

**On the weighting bug.**
We divided the weights by the number of output HDF5 files.  We should have
divided by the number of input Level 3 files.  We put ten input files into one
output file, so the rate was wrong by a factor of a hundred.  The comparison with
Table 13 found it.  This is the argument for always doing that comparison.

**On the pulse series behind `micro_count`.**
The reference says the cleaned series.  The original pass2 code starts from the
uncleaned one.  We follow the reference.  We also measured the difference, and it
is very small.  The two chains give the same count in about ninety percent of
events.  The 200 nanosecond window decides the result, and noise hits are spread
over ten microseconds anyway.

**On the 0.70 cut.**
It comes from the reference.  We can use it directly, because LightGBM gives a
probability.  We did not change it, and we should not change it yet.  The Level 4
cut uses both classifiers together, and the muon one does not exist yet.

---

# Note for the speaker, not part of the talk

We made the HDF5 files for these numbers before the `micro_count` cleaning fix.
The effect of that fix is small.  The count is the same in about ninety percent
of events.  But a new processing run before the talk means the numbers on slides
10 and 12 need a re-check.  They may not stay the same.
