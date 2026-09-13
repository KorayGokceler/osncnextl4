# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides.  About 3,500 spoken words, so around 25 minutes at a normal
speed.  The times add up from the start.

**If you only have 15 minutes,** cut in this order and nothing breaks.  Drop
slide 8 completely; the audience knows what a BDT is, so move its output-scale
sentence into slide 9.  Then drop the first half of slide 9 and keep only the
table and the gap to the star.  Then drop the left-panel walk-through on slide
12 and keep the right panel.  That saves about eight minutes.  Please do not
cut slide 4, slide 7, or the first item on slide 13.  Those three carry the
thread.

Two things to know before you read it.

**The talk has one thread.**  Level 4 must separate signal from background with
cheap variables only.  I had to write five of those variables myself.  So two
questions run through the whole talk: *are my variables right?* and *is my
classifier good enough?*  Each slide moves one of them forward.  The links
between slides are written into the script, so please do not cut them.

**Every plot gets the same treatment.**  First the axes.  Then the colours.
Then one place to look.  Then what it means.  The audience cannot read a plot
and listen at the same time, so tell them where to put their eyes.

The English is simple on purpose.  Short sentences, one idea each.

---

## Slide 1 — Title  ·  0:00–0:50

Thank you.  I have been rebuilding Level 4 of the oscNext selection for pass3.

Level 4 is the step that removes noise and atmospheric muons.  I had to write
part of it again from scratch.  So this talk answers two questions.  First: did
I write the variables correctly?  Second: is the classifier good enough?

I will give you the short answer now.  The classifier is good enough.  The
variables are probably correct, but I cannot prove it yet.  The rest of the talk
is the evidence for both.

One note on sources.  When I say "the reference", I mean the oscNext note,
*Simulations and sample*, version 00.07.  I did not process pass2 myself.  Every
pass2 number in this talk comes from that document.

---

## Slide 2 — The oscNext selection chain  ·  0:50–2:30

Let me start with where Level 4 sits, because that explains its constraints.

This table shows the chain.  These are the words of the reference, section 1.3.
Read down the middle column.  Level 2 is the collaboration filter.  Level 3 uses
simple, fast cuts, and its job is to make the data and the MC agree.  They must
agree before machine learning is possible at all.  Level 4 is that machine
learning.  Level 5 removes muons that sneak in through the corridors.  Level 6
does the full reconstruction.  Level 7 cuts on the reconstruction.

Now look at the order.  Each level is cheaper than the next one.  That is the
whole design.  And it gives Level 4 a hard rule: **Level 4 may not reconstruct
anything.**  Reconstruction is expensive, and Level 6 pays for it.  Level 4 gets
only fast variables.

Please keep that rule in mind.  It is the reason the rest of this talk is about
fourteen simple numbers instead of about a fit.

The small table at the bottom shows why Level 4 matters.  At Level 3 the muon
rate is about 505 millihertz.  Noise is about 37.  Neutrinos are about 5.  So
background is a hundred times larger than signal.  Level 4 has to turn that
around, with cheap variables only.

*(Only if someone asks: the sample splits in two at Level 6.  One part is a
verification sample, the other has higher statistics.  So there are really two
L6 stages and two L7 stages.)*

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  2:30–4:30

So how does Level 4 do it?  With two classifiers.  One removes pure noise.  One
removes atmospheric muons.  An event passes if the noise score is 0.70 or higher
**and** the muon score is 0.65 or higher.  Both cut values come from the
reference, section 3.5.

Now the problem.  The original Level 4 code still exists, but somebody commented
out the whole body.  It also needs three projects, and our meta-project does not
have any of them.  `icecube.oscNext` applies the classifier.  `tau_bdt` gives
VICH.  `analysis.event_selection` gives the Dunkman variables.  So we cannot
simply switch the old code on.  I had to build the variables again.

Look at the table on the right.  There are fourteen variables in total.  Four
come ready from Level 3, so there is no work there.  Five come from IceTray
projects that we do have.  And I wrote five of them again myself, in plain
Python, from the descriptions in the reference.

The red row is the point of this slide.  All the risk sits in those five.  And
here is the reason I keep coming back to them.  If you compute a variable the
wrong way, you do not get an error.  You get a number.  The job runs, the plots
look fine, and the classifier trains happily on wrong data.

So I compared my code with the reference, line by line.  I also compared it with
the original pass2 code, because we still have that file even though it does not
run.  The comparison found bugs.  Every one of them gave wrong numbers, and none
of them gave an error.  You only find bugs like this if you go looking.

That comparison raised a second problem, and it takes us to the next slide.

*(Only if someone asks.  Do not list them first.  An index table was writing
over the data.  Event IDs are not unique between files, so some rows went to the
wrong event.  A flux event count was read once and then reused for ten files.
And one noise cleaning step in `micro_count` was skipped.  That last bug comes
from the original pass2 code, not from us.  The original author even left a
comment there and asked if that series is used at all.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  4:30–6:00

Here is the second problem.  To check my code against the reference, the
reference has to define the variables.  For two of them, it does not.

This table has the five noise inputs.  The descriptions come from Table 11, word
for word.  I show their exact wording on purpose, so you can see the gaps
yourself.

Look at the fourth row, `fill_ratio`.  It says "measure of the geometrical
spread of the hits about some vertex".  Then it says "details here", in red.
That is an empty cross-reference in the document.  Nobody filled it in.  So the
reference never tells us which vertex to use.  Please remember this row.  It
comes back on slide 7, and it becomes the most important thing in the talk.

Now the last row, `FullTimeLengthRatio`.  It tells us this is a ratio of the
cleaned and the uncleaned duration.  It does not tell us which one goes on top.

I found both answers outside the document.  For the vertex, I read the original
pass2 code: it uses the position of the first HLC hit.  For the direction of the
ratio, I used Figure 13: its axis runs from zero to one, and only cleaned over
uncleaned can do that.

I want to be open about this.  This is the weakest part of the chain.  I took
these answers from a code file and from a figure axis.  Nobody wrote them down
properly.

So the variables are built, and two of them rest on guesses.  The next question
is simple.  What do they actually look like?

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  6:00–8:00

This is the answer.  These are the five inputs, computed on our own pass3 files.

Let me give you the axes first.  Each panel is one variable.  The x axis is the
variable itself.  The y axis is rate, in hertz per bin, on a log scale.  Blue is
neutrino.  Red is noise.

Start with the top left panel, `NchCleaned`, the number of hit DOMs.  The red
curve dies quickly as you move right.  The blue curve keeps going out to a
hundred and forty.  So noise events are small and neutrino events can be large.
That is the behaviour we expect, and it is reassuring.

The top middle panel, `micro_count`, does the same thing in the same direction.
The top right panel is `iLineFit_speed`, on a log x axis.  Notice that the two
curves sit almost on top of each other.  Keep that in mind for slide 7.

Now the bottom right panel, `FullTimeLengthRatio`.  This one surprised us, and I
want to spend a minute on it.

Our own docstring used to say something about this variable.  It said the value
is close to one for a real event, and close to zero for noise.  That is simply
wrong.  We measured it, and now we understand why.

The uncleaned series is `SplitInIcePulses`.  It covers the whole readout window,
about ten microseconds.  It does that in every event, signal or noise.  So the
bottom of the ratio is almost a constant.  The variable is really just the
cleaned duration, divided by ten microseconds.

Look at the two curves in that panel.  The blue one peaks at a lower value than
the red one.  The median is about 0.16 for electron neutrinos and about 0.27 for
noise.  So the noise events have the **longer** cleaned series, not the shorter
one.

The variable still separates well.  But the reason is the opposite of what we
wrote down.  A low-energy cascade is short in time.  Noise is not long.  We had
the right variable and the wrong story.  If we had believed the docstring, we
would have read this plot backwards.

So the variables behave sensibly.  That leads to the next question.  Are they
five different variables, or one variable five times?

---

## Slide 6 — Input correlations  ·  8:00–9:15

These two matrices answer that.

Both axes list the five inputs, in the same order.  The colour is the Spearman
rank correlation.  Dark red is plus one, dark blue is minus one, white is zero.
The number is printed in each box.  Signal is on the left, background on the
right.  I keep them separate, because a pair can be correlated in one class and
not in the other.

The diagonal is dark red in both, and that is just each variable against itself.
Ignore it.

Now look away from the diagonal.  Almost everything is pale.  On the signal
side, the strongest pair is `NchCleaned` with `micro_count`, at 0.71.  That makes
sense, because both of them count hit DOMs.  Everything else is small.  On the
background side, even that pair drops to 0.10.

So the answer is no.  These are five different variables.  Nothing here is a
copy of something else.

This matters for the next slide, so let me say it clearly.  If one of these
variables turns out to add almost nothing to the model, we now know how to read
that.  The variable is weak.  It is not weak because another variable already
covers it.

---

## Slide 7 — Feature importance  ·  9:15–11:00

So which variable does the work?  This plot answers it, and it is the most
important plot in the talk for our next steps.

The y axis lists the five inputs, sorted from top to bottom.  The x axis is gain,
in percent.  Gain means how much each variable improved the splits during
training.  The number is printed at the end of each bar.

Look at the top bar.  `fill_ratio` is about sixty percent.  It is more than twice
the next one.  `NchCleaned` is twenty-six percent.  Then `FullTimeLengthRatio` at
eight, `micro_count` at four, and `iLineFit_speed` at one percent.

Two things follow from this.

First, the bottom bar.  `iLineFit_speed` is almost dead.  And after the last
slide, we know what that means.  It is not hidden inside another variable.  It
simply does not separate.  You saw that already on slide 5: its blue and red
curves sat on top of each other.

Second, and this is the important one, look at the top bar again.  The model
leans on `fill_ratio` more than on everything else together.  Now go back to
slide 4.  `fill_ratio` is the variable with the empty cross-reference.  The
reference never tells us where its vertex is.

And it gets worse.  `fill_ratio` has one free parameter, the spherical radius.
Its value is 1.6, and I took it from the original code.  The comment next to that
value says somebody tuned it for GRECO.  Nobody tuned it again for oscNext.

So our model depends most on a parameter from a different event selection.  That
is a risk, but it is also an opportunity.  Re-tuning that number is the cheapest
improvement available to us.  It is one parameter and one scan.  It is the second
item on my next-steps slide.

That closes the variables.  Now the classifier.

---

## Slide 8 — How a boosted decision tree works  ·  11:00–11:50

This will be short, because you all know it.  I need it only for one detail.

One shallow tree is a weak classifier.  Boosting trains many trees, one after
another.  Each new tree corrects the mistakes of the earlier ones.

The two engines pick the next tree in different ways.  AdaBoost gives more weight
to the events it gets wrong.  Gradient boosting fits the next tree to the
gradient of the loss.  The reference uses LightGBM, section 3.6.1, and LightGBM
is gradient boosting.

Here is the detail I need.  It is the output scale.  LightGBM gives you a
probability, between zero and one.  So the cut value 0.70 from the reference
works on our model directly.  A pybdt score is not on that scale.  With pybdt,
the threshold from the reference means nothing, and you have to find your own.

Keep that in mind.  It is part of the price on the next slide.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  11:50–13:45

We trained with pybdt first.  That is IceCube's own BDT library, and it uses
AdaBoost.  We chose it instead of the reference method, and the argument was
simple: pybdt is our in-house tool.

Let me give you this plot carefully, because we will see the same axes again on
the next slide.

The x axis is background rejection, in percent.  It starts at 80 and ends at 100.
Good is to the right.  The y axis is signal efficiency, also in percent.  Good is
up.  So the best place on this plot is the top right corner.  Each coloured curve
is one configuration.  The black star is the target from the reference.

Now look at the shape.  On the left side, at 80 percent rejection, the curves sit
high, near 95.  That part is easy.  Then follow them to the right.  After about
97 percent they all fall off a cliff.  And the star sits above every curve, in
empty space.

That gap is the whole story of this slide.  The table gives you the numbers.  The
best configuration keeps 94 percent of the signal at 90 percent rejection.  At 99
percent rejection it keeps only 66.  The target is about 96.  So we were thirty
points short at the point that matters.

Two remarks before I move on.

First, one good property.  The training is deterministic.  Run the same setup
twice and you get exactly the same scores.  So we never have to average, and we
never have to worry about a seed.

Second, a warning about method.  We used the Kolmogorov–Smirnov p-value to test
for overtraining, and on this data it does not work.  It said our best model was
overtrained, with p equal to 0.002.  It said the worst model was fine, with p
equal to 0.98.  The reason is simple.  We have about a hundred thousand signal
events, and with so many events KS finds differences that are real in statistics
and meaningless in physics.  We replaced it with a simpler test.  We take the
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

I changed one thing only.  Same events, same five variables, same train and test
split, same weights, same scoring code.  Only the trainer is different.  That is
important, because it means this plot is a fair comparison.

The axes are the same as before.  Background rejection on the x axis, signal
efficiency on the y axis, target star at the top right.  The green solid curve is
LightGBM.  The two dashed curves are the best pybdt models from the last slide.

Now compare the shapes.  The dashed curves start to fall at about 95 percent.
The green curve stays flat, almost at a hundred, all the way to 99.  It only
turns down right at the end.  And the green curve passes through the star.

The table has the numbers.  At 90 percent rejection, 94 becomes 99.  At 95
percent, 91.7 becomes 98.5.  And at 99 percent rejection, 65.8 becomes 95.9.
That reaches the target of the reference.

Two things make me believe this result.

First, there is no overtraining.  The train and test efficiencies differ by one
tenth of a point.  I will show you that directly on the next slide.

Second, the model reached the target with a handicap.  The reference asks for at
least 500 events in one leaf.  Somebody tuned that number on much more noise
simulation than we have.  We only had about 830 background events in the fit, so
that setting limits the trees very hard.  The training also stopped early, at 122
trees out of 2,000.  The model still reached the target.

And this result changed our minds about something.  Before this, we thought our
background statistics were the problem.  We thought the model could not do
better, because we did not have enough noise MC.  This measurement showed that we
were wrong.  The problem was the engine, not the statistics.

That was the claim.  Here is the proof.

---

## Slide 11 — LightGBM score distribution  ·  15:30–16:30

This is the overtraining check.

The x axis is the classifier output, from zero to one.  The y axis is the
training weight, added up in each bin.  There are four curves.  Two are signal,
train and test.  Two are background, train and test.  Both panels show the same
histogram.  The left one is linear.  The right one is on a log scale.

Look at the left panel first.  Signal piles up against one, on the right.
Background piles up against zero, on the left.  The middle is nearly empty.  That
is what a working classifier looks like.

But the left panel cannot answer the overtraining question, and this is why we
need the right one.  Overtraining would show up in the tail between the two
peaks.  On a linear axis that tail is flat on the floor.  On the log axis you can
see it.

So look at the right panel now, in the middle region.  The train curve and the
test curve follow each other bin by bin.  They wander together, they do not
separate.  That is the check, and it passes.

Two more things to notice.

Please read the y axis carefully.  We normalise each class to the same total.  So
the blue area and the red area are equal by definition.  Please do not read this
as "we have as much background as signal".  We do not.  For noise the weights are
all equal, so the red curves are really event counts.

And look at where the red curves stop, on the right panel.  They stop around
0.85.  No background event in the test set gets a higher score than that.  Keep
that number in mind for the next slide.

---

## Slide 12 — The limit: noise MC statistics  ·  16:30–17:45

If the engine was the problem, and we fixed it, what limits us now?

This plot has two panels.  Let me take the left one first.  The x axis is the cut
value on the classifier output.  The y axis is the percentage of events that
survive.  The blue curve is the signal we keep.  The red curve is the background
we reject.  The dotted vertical line is the cut value from the reference, 0.70.

Follow the red curve up from zero.  It climbs almost vertically, and by 0.2 it is
already near a hundred.  The blue curve stays flat at a hundred until about 0.85.
So between those two values we reject almost all the noise and lose almost no
signal.  The reference cut at 0.70 sits inside that window.  That is a comfortable
place to be.

Now the right panel.  Same axes as slides 9 and 10: rejection across, efficiency
up, target star.  The grey vertical lines are the new information.  They mark
where the background statistics run out.  They sit at a hundred, ten and one
remaining background events.

Look at where the star sits, relative to those grey lines.  It is past the "ten
events" line.

That is the limit, and the numbers are simple.  Our test set has about 330,000
signal events and 2,051 noise events.  That is 159 to one.  At 90 percent
rejection, about 100 background events survive the cut.  At 95 percent, 50.  At
99 percent, ten.  At 99.9 percent, one.  The target of the reference sits at 99.2
percent rejection, and only eight events define that point.

So I want to be careful about what I ask for.  We have enough noise MC to reach
the target.  Slide 10 showed that.  But we do not have enough to measure anything
above 99 percent rejection.  Fifty events support the 95 percent row, so that row
is solid.  Ten events support the 99 percent row, so that row is at the edge.

More vuvuzela simulation is useful and I would like to have it.  But we need it
to measure the far tail, not to reach the target.  It does not block the
classifier.

---

## Slide 13 — Status and next steps  ·  17:45–19:00

Let me pull the two threads together.

On the classifier, the answer is yes.  Level 3 to Level 4 is rebuilt for pass3
and it runs reliably.  The pass3 production contains some half-written files, and
one bad file can kill a whole job, so the code scans the files first and skips
them.  We found all fourteen inputs and fixed their column names.  And the noise
classifier reaches the target of the reference.

On the variables, the answer is "probably, but not proven".  And that is the
first item here.

We should compare our variables with the pass2 Level 4 files.  Those files
already contain these variables, and the original code produced them.  So we can
compare mine with theirs, event by event.  This is the only real way to check the
five variables I wrote again.  Everything I told you about them today came from
reading a code file and reading a figure axis.  This test would replace that with
a measurement.  It would also let us train on one pass and test on the other one.
That is a much stronger test.  So if you know where those files are, please
tell me after the talk.

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

# If someone asks

**"Why not just use the original Level 4 code?"**
Somebody commented out the whole body, and it needs three projects that we do not
have.  We keep the file, and it helped us a lot with parameter values.  But it
does not run.

**"How do you know your rewritten variables are correct?"**
I do not know it completely.  That is why it is the first item on my list.  But I
can say three things.  I compared them with the reference, line by line.  I
compared them with the original code, and that found four real bugs.  And the
rates agree with Table 13.  A direct comparison with pass2 L4 files would give us
the real answer.

**"Is 95.9 percent meaningful with only ten background events?"**
That is a fair question.  Ten test-set background events define the 99 percent
point, so the uncertainty there is large.  The 95 percent row has fifty events,
so it is much better.  There LightGBM gives 98.5 and AdaBoost gives 91.7.  So the
order is clear.  The exact value at 99 percent is not.

**"Why does the x axis on slide 12 go past one?"**
Good catch.  The cut scan adds two points outside the real range, so the curve
closes at both ends.  Only the region between zero and one is a real cut.  The
straight fall on the far right is a line between two points, not a measurement.

**"Why is your signal rate higher than Table 13?"**
Our flux is a simple power law.  It is not a real atmospheric flux with
oscillations.  A factor of about 1.5 is normal for that.  The noise rate does not
depend on this choice, and it agrees to 13 percent.  That tells us the vuvuzela
weight unit in pass3 is correct.

**"What was the weighting bug?"**
We divided the weights by the number of output HDF5 files.  We should have
divided by the number of input Level 3 files.  We put ten input files into one
output file, so the rate was wrong by a factor of a hundred.  The comparison with
Table 13 found it.  This is why we always do that comparison.

**"Which pulse series does `micro_count` start from?"**
The reference says the cleaned series.  The original pass2 code starts from the
uncleaned one.  We follow the reference.  We also measured the difference, and it
is very small.  The two chains give the same count in about ninety percent of
events.  The 200 nanosecond window decides the result, and noise hits are spread
over ten microseconds anyway.

**"Is the 0.70 cut yours or the reference's?"**
It comes from the reference.  We can use it directly, because LightGBM gives a
probability.  We did not change it.  And we should not change it yet.  The Level 4
cut uses both classifiers together, and the muon one does not exist yet.

---

# Note for the speaker, not part of the talk

We made the HDF5 files for these numbers before the `micro_count` cleaning fix.
The effect of that fix is small.  The count is the same in about ninety percent
of events.  But if you process the files again before the talk, please check the
numbers on slides 10 and 12 again.  Do not assume they stay the same.
