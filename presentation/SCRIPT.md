# Speaking script — oscNext Level 4: rebuilding the noise classifier

Thirteen slides.  About 2,000 spoken words.  That is around 15 minutes at a
normal speed, so there is time for questions.  The times add up from the start.

The English here is simple on purpose.  Short sentences.  One idea in each one.
No difficult grammar.  Every number is already on a slide or in a plot, so you
do not need to learn anything by heart.  Point at the plot and say the line.

---

## Slide 1 — Title  ·  0:00–0:45

Thank you.  I have been rebuilding Level 4 of the oscNext selection for pass3.

I want to show you three things today.  First, my processing gives the same
rates as the reference document.  Second, the noise classifier now works as well
as the reference says it should.  Third, some parts are still not checked.

One note before I start.  When I say "the reference", I mean the oscNext note,
*Simulations and sample*, version 00.07.  I did not process pass2 myself.  Every
pass2 number here comes from that document.

---

## Slide 2 — The oscNext selection chain  ·  0:45–2:15

This is the selection chain.  These are the words of the reference, from
section 1.3.

Look at the shape of it.  Level 2 is the collaboration filter.  Level 3 uses
simple, fast cuts.  Its job is to make the data and the MC agree.  They have to
agree before we can use machine learning at all.  Level 4 is that machine
learning.  Level 5 removes muons that come in through the corridors in the veto.
Level 6 does the final reconstruction.  Level 7 cuts on it.

So each level is cheaper than the next one.  That is the design.  And this is the
reason Level 4 cannot reconstruct anything.  Reconstruction costs too much, and
Level 6 pays for it.  Level 4 can only use fast variables.

Level 4 has a lot of work to do.  This table is Table 13.  At Level 3 the muon
rate is about 505 millihertz.  The noise rate is about 37.  The neutrino rate is
about 5.  So background is a hundred times larger than signal.  Level 4 turns
this around.

*(Only if someone asks: the sample splits in two at Level 6.  One is a
verification sample, the other has higher statistics.  So there are really two
L6 stages and two L7 stages.)*

---

## Slide 3 — What L4 does, and what had to be rebuilt  ·  2:15–4:00

Level 4 uses two classifiers.  One removes pure noise.  One removes atmospheric
muons.  An event passes if the noise score is 0.70 or higher, and the muon score
is 0.65 or higher.  Both cut values come from the reference, section 3.5.

So why did I rebuild it?  The original Level 4 code is still there, but somebody
commented out the whole body.  It also needs three projects, and our
meta-project does not have them.  `icecube.oscNext` applies the classifier.
`tau_bdt` gives VICH.  `analysis.event_selection` gives the Dunkman variables.
We have none of these.  So we cannot simply turn the old code on.

This means I had to make the variables again.  There are fourteen of them.  Four
come ready from Level 3, so there is no work there.  Five come from IceTray
projects that we do have.  And I had to write five of them again, in plain
Python, from the descriptions in the reference.

All the risk sits in that last group.  Those five can be wrong.  And they can be
wrong quietly.  If you compute a variable the wrong way, you do not get an
error.  You just get a number.

So I compared my code with the reference, line by line.  I also compared it with
the original pass2 code.  We do have that file, even if it does not run.

The comparison found bugs.  All of them gave wrong numbers, and none of them
gave an error.  You only find bugs like this if you look for them.

*(Only if someone asks.  Do not list them first.  An index table was writing
over the data.  Event IDs are not unique between files, so some rows went to the
wrong event.  A flux event count was read once and then used for ten files.  And
one noise cleaning step in `micro_count` was skipped.  That last bug comes from
the original pass2 code, not from us.  The original author even wrote a comment
there and asked if that series is used at all.)*

---

## Slide 4 — Noise BDT inputs, as the reference defines them  ·  4:00–5:30

These are the five inputs of the noise classifier.  The descriptions come from
Table 11 of the reference, word for word.

I show them as quotes for a reason.  Two of them are not really definitions.

Look at `fill_ratio`.  It says "measure of the geometrical spread of the hits
about some vertex".  Then it says "details here".  That is an empty
cross-reference in the document.  It never tells us where the vertex is.  Please
remember this.  I come back to it in three slides.

Now look at `FullTimeLengthRatio`.  It says this is a ratio of the cleaned and
the uncleaned duration.  But it does not say which one goes on top.

I found both answers outside the document.  The vertex comes from the original
pass2 code.  That code uses the position of the first HLC hit.  The direction of
the ratio comes from Figure 13.  Its axis goes from zero to one.  Only cleaned
over uncleaned can give you that.

I want to be open here.  This is the weakest part of the chain.  I took these
answers from a code file and from a figure.  Nobody wrote them down properly.

---

## Slide 5 — Noise BDT inputs, signal vs. noise  ·  5:30–7:00

These are the five inputs on our own pass3 files.  Signal is blue and noise is
red.  This is rate against the variable.

They separate.  That is the first thing you want to see.  But one of them
separates in the opposite direction.  I think this is worth a minute.

Our own docstring used to say something about `FullTimeLengthRatio`.  It said the
value is close to one for a real event and close to zero for noise.  That is
simply wrong.  We measured it, and now we know why.

The uncleaned series is `SplitInIcePulses`.  It covers the whole readout window.
That is about ten microseconds.  It is the same in every event, signal or noise.
So the bottom of the ratio is almost a constant number.  The variable is really
just the cleaned duration, divided by ten microseconds.

The median is about 0.16 for electron neutrinos.  For noise it is about 0.27.  So
the noise events have the longer cleaned series, not the shorter one.  The
variable still separates well.  But the reason is different.  A low-energy cascade
is short in time.  Noise is not long.  The variable did the right thing for the
wrong reason.  If we had trusted the docstring, we would have read this plot
backwards.

---

## Slide 6 — Input correlations  ·  7:00–8:00

Before training, there is one obvious question.  Do these five variables say the
same thing five times?

These are Spearman rank correlations.  Signal on the left, background on the
right.  I compute them separately.  A pair can be correlated in one class and not
in the other one.

The answer is no.  They are mostly independent.  Nothing here is a copy of
something else.  Every input brings something new.

So if one of them adds almost nothing, we know what that means.  The variable is
weak.  It does not mean that another variable already covers it.  Please keep
this in mind for the next slide.

---

## Slide 7 — Feature importance  ·  8:00–9:15

This is the gain split of the trained noise model.  For our next steps, this is
the most important plot in the talk.

`fill_ratio` carries about sixty percent of the gain.  `NchCleaned` carries about
twenty-six percent.  `FullTimeLengthRatio` about eight.  `micro_count` about
four.  And `iLineFit_speed` about one percent.  That last one is almost dead.

Now put this together with slide 4.  The model uses `fill_ratio` more than
anything else.  But the reference never tells us where its vertex is.  And it has
one free parameter, the spherical radius.  Its value is 1.6, and it comes from
the original code.  The comment next to it says something important.  Somebody
tuned this value for GRECO, and nobody tuned it again for oscNext.

So the model leans on a parameter from a different selection.  Re-tuning it is
the best next step we have.  And it is cheap.  It is one number and one scan.

---

## Slide 8 — How a boosted decision tree works  ·  9:15–10:00

This will be short, because you all know it.  One shallow tree is a weak
classifier.  Boosting trains many trees, one after the other.  Each new tree
corrects the mistakes of the earlier ones.

The two engines choose the next tree in different ways.  AdaBoost gives more
weight to the events it gets wrong.  Gradient boosting fits the next tree to the
gradient of the loss.  The reference uses LightGBM, in section 3.6.1, and that is
gradient boosting.

One detail matters later.  It is the output scale.  LightGBM gives you a
probability, between zero and one.  So the cut value 0.70 from the reference
works on our model directly.  A pybdt score is not on that scale.  With pybdt you
have to find a new threshold yourself.

---

## Slide 9 — Training with pybdt (AdaBoost)  ·  10:00–11:45

We trained with pybdt first.  That is IceCube's own BDT library, and it uses
AdaBoost.  We chose it instead of the reference method.  The argument was simple:
pybdt is our in-house tool.

This plot shows six configurations.  It is signal efficiency against background
rejection, on the test set.  The table gives you three working points.

The best one keeps about 94 percent of the signal at 90 percent rejection.  But
at 99 percent rejection it keeps only 66 percent.  The reference target is about
96 percent.  So at the important working point, we were thirty points too low.

Two remarks here.

First, one good property.  The training is deterministic.  Run the same setup
twice, and you get exactly the same scores.  So we do not need to average
anything, and we do not need to worry about a seed.

Second, a warning about method.  We used the Kolmogorov–Smirnov p-value to test
for overtraining.  On this data it does not work.  It said our best model was
overtrained, with p equal to 0.002.  And it said the worst model was fine, with p
equal to 0.98.  The reason is simple.  We have about a hundred thousand signal
events.  With so many events, KS finds differences that are real in statistics
but mean nothing in physics.  Now we use a better test.  We look at the
efficiency on the train set and on the test set, at the same rejection, and we
take the difference.

I should also be fair here.  We chose these hyperparameters by hand.  We did not
tune them as carefully as the reference did.  So I will not say "AdaBoost is a
worse algorithm".  I will say "our pybdt setup was costing us a lot".

---

## Slide 10 — Training with LightGBM  ·  11:45–13:15

So we went back to the method of the reference.  LightGBM, with the
hyperparameters from Table 10.

Same events.  Same five variables.  Same train and test split.  Same weights.
Same code for the scoring.  Only the trainer is different.

At 90 percent rejection, 94 becomes 99.  At 95 percent, 91.7 becomes 98.5.  And
at 99 percent rejection, 65.8 becomes 95.9.

This reaches the target of the reference.

Two things make me believe it.

First, there is no overtraining.  The train and test efficiency differ by one
tenth of a point.  And the two score distributions sit on top of each other.  You
will see that on the next slide.

Second, the model reached the target with a handicap.  The reference asks for at
least 500 events in one leaf.  Somebody tuned that number on much more noise
simulation than we have.  We only had about 830 background events in the fit.  So
this setting limits the trees a lot.  The training also stopped early, at 122
trees out of 2,000.  The model still reached the target.

I want to be clear about one thing.  This result changed our minds.  Before, we
thought our background statistics were the problem.  We thought the model could
not do better, because we did not have enough noise MC.  This measurement showed
that we were wrong.  The problem was the engine.

---

## Slide 11 — LightGBM score distribution  ·  13:15–14:00

This is the overtraining check.  The left panel is linear.  The right panel shows
the same histogram on a log scale.

We need the log panel.  Overtraining would show up in the tail between the two
peaks, and you cannot see that tail on a linear axis.

Train and test sit on top of each other in both panels.

Please read the y axis carefully.  It is the training weight, added up in each
bin.  We normalise each class to the same total.  So the two areas are the same
by definition.  Please do not read this as "we have as much background as
signal".  For noise the weights are all equal, so the background curves are
really event counts.

And look at the background.  It stops around 0.85.  No background event in the
test set gets a higher score than that.

---

## Slide 12 — The limit: noise MC statistics  ·  14:00–15:15

If the engine was the problem, what is the limit now?

The test set has about 330,000 signal events and 2,051 noise events.  That is 159
to one.  And the cut removes the noise very fast.  At 90 percent rejection, about
100 background events stay above the cut.  At 95 percent, 50 stay.  At 99
percent, ten.  At 99.5 percent, five.  At 99.9 percent, one.

The target of the reference is at 99.2 percent rejection.  Only eight events in
our test set define that point.

So I want to be careful with my request.  We have enough MC to reach the target.
The last slides showed that.  But we do not have enough MC to measure anything
above 99 percent rejection.  Fifty events support the 95 percent row, so that row
is solid.  Ten events support the 99 percent row, so that row is at the edge.

More vuvuzela simulation is useful, and I would like to have it.  But we need it
to measure the far tail.  It does not block the classifier.

---

## Slide 13 — Status and next steps  ·  15:15–16:30

Let me summarise.

Level 3 to Level 4 is rebuilt for pass3, and it runs reliably.  The pass3
production has some half-written files, and one bad file can kill a whole job.
So the code scans the files first and skips the bad ones.  We found all fourteen
inputs and fixed their column names.  We checked the rewritten variables against
the reference and against the original code.  And the noise classifier is trained
and reaches the target.

Now the next steps, in order of value.

The first one is where I need help.  We should compare our variables with the
pass2 Level 4 files.  Those files already have these variables, and the original
code computed them.  We can compare mine with theirs, event by event.  This is
the only real way to check the five variables I rewrote.  Everything I said about
them today comes from reading code and reading a figure.  We could also train on
one pass and test on the other one.  That test is much stronger.  So if you know
where those files are, please tell me after the talk.

The second step is the `fill_ratio` radius, for the reason on slide 7.

The third step is the muon classifier.  The CORSIKA background is ready.  And we
already split train and test by shower, so copies of the same air shower cannot
go to both sides.

The fourth step is more vuvuzela simulation, to measure the far tail.

The fifth step is to process the rest of the samples.  Corrupt input files
stopped the muon-neutrino and noise jobs early.

Two limits, and I want to say them clearly.  We have no tau neutrino set here,
and that is a few percent of the signal.  And our muon background is CORSIKA, not
real data.  So we cannot compare data and MC yet.

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
