# Assigning colours from channel intensities

How `Analysis/Dual-Color/Assign Colors` turns two photon counts into a species
label.  The first half is the histogram-and-minimum rule everybody uses; the
second half is the probabilistic rule, which is what the *allowed crosstalk*
parameter needs to mean something.  Decisions taken 2026-09-14.

## The observable

A ratiometric two-colour experiment splits each molecule's emission between two
detection channels and reads the split.  The dual-channel global fit gives, per
localization, the photons in each channel -- `photons_ch0`, `photons_ch1` --
with x, y and z shared (`dualfit.LINK_XYZ`): the photon split is deliberately
left free, because that split *is* the colour.

Write the two counts N1, N2, the total N = N1 + N2, and

    r = (N1 - N2) / (N1 + N2)   in [-1, 1].

r is the natural coordinate rather than N1/N2 or a bare fraction: it is
bounded, symmetric between the channels, and -- see below -- its noise has a
closed form that is the same at both ends.  A species k has a characteristic
true ratio rho_k, and the histogram of r over a mixed sample shows one mode per
species.

## Mode 1: split at the minima

Histogram r, smooth it (a Gaussian of `smoothing` bins, so that shot noise in
the bins does not make its own maxima), take the `colors` highest local maxima
as the species' rho_k, and put the boundary at the lowest point of the smoothed
histogram between neighbouring maxima.  A localization is assigned to the
interval it falls in, 1 to n from left to right, except within +-dr of a
boundary, where it is left at 0.

This is a decision on r alone.  Its weakness is exactly that: a localization
with 5000 photons measures r to about 0.014, and one with 200 photons to about
0.07, and the fixed dr has to be set for the dim ones or it throws away
everything.  The exclusion zone is a blunt instrument standing in for a
per-localization confidence.

## The noise on r

Photon counting is Poisson, so under species k with true ratio rho,

    mu_1 = N (1 + rho) / 2,   mu_2 = N (1 - rho) / 2.

Propagating independent errors sigma_1, sigma_2 through r (dr/dN1 = 2 N2/N^2,
dr/dN2 = -2 N1/N^2), evaluated at the hypothesis:

    sigma_r^2 = [ (1 - rho)^2 sigma_1^2 + (1 + rho)^2 sigma_2^2 ] / N^2.

With pure shot noise, sigma_i^2 = mu_i, this collapses to

    sigma_r^2 = (1 - rho^2) / N.

which is exact rather than approximate: N1 ~ Binomial(N, (1+rho)/2) has
variance N p (1-p), and r = 2 N1/N - 1 carries that to 4 p (1-p) / N =
(1 - rho^2)/N.  Two things follow that matter for the implementation.  The
precision of r improves as 1/sqrt(N), so the boundary should move with
brightness.  And it improves towards the ends of the axis -- a molecule that
emits almost everything into one channel is measured more sharply than one that
splits evenly -- so a symmetric fixed dr is the wrong shape even for one
brightness.

The fit's own `photons_err_ch*` (the CRLB) are used when the table has them,
because they also carry the background under the PSF and the EMCCD excess-noise
factor, both of which make a localization less informative than its raw photon
count suggests.  They enter through an **effective photon number**: take the
propagated sigma_r at the measured point,

    sigma_r^2 = 4 (N2^2 sigma_1^2 + N1^2 sigma_2^2) / N^4,
    N_eff     = (1 - r^2) / sigma_r^2,

and then use the shot-noise form with N_eff in place of N.  N_eff is what the
localization is worth in photons: it equals N when the errors are exactly
Poisson, and is smaller when background or EM gain has cost information.  It
keeps the rho-dependence of the variance, which a plug-in sigma_r would throw
away -- and which matters, since a plug-in variance goes to zero as |r| -> 1
and would make the dimmest, most extreme localizations look the most certain.
Without fitted errors, N_eff = N: the sqrt(photons) fallback.

Real modes are usually broader than this, because the dye's split is not a
single number (spectral heterogeneity, a fluorophore's environment, an
imperfect splitter).  `spread` adds a species-intrinsic sigma in quadrature,

    s_k^2 = (1 - rho_k^2) / N_eff + spread^2,

and the preview prints the observed mode width next to the median shot-noise
width so it can be set by looking rather than by guessing.  It is 0 by default:
shot noise only.

## The probability of each colour, derived

The question, exactly as it should be asked: *the expected ratios rho_c are
known (from the histogram maxima, or measured); this molecule gave I1 and I2
photons; what is P(colour 1) and what is P(colour 2)?*

Photons arrive as counts, so start there rather than at r.  A molecule of
species c whose emission is collected with expected total lambda sends each
detected photon to channel 1 with probability

    p_c = (1 + rho_c) / 2,

and the two detector counts are independent Poissons,

    I1 ~ Poisson(lambda p_c),   I2 ~ Poisson(lambda (1 - p_c)).

**The brightness cancels.** A pair of independent Poissons factorizes into
their total and their split,

    P(I1, I2 | lambda, c) = Poisson(N | lambda) * Binomial(I1 | N, p_c),
    N = I1 + I2,

and the two factors separate what we do not know from what we want: the total
depends on lambda and not on the colour, the split on the colour and not on
lambda.  Marginalizing over an unknown brightness distribution f(lambda)
therefore leaves that unknown entirely in the first factor,

    P(I1, I2 | c) = [ integral Poisson(N | lambda) f(lambda) d lambda ]
                    * Binomial(I1 | N, p_c),

and the bracket -- whatever it is, and it is the hard part to model -- is the
same for every colour, so it cancels in Bayes' rule.  Nothing about how bright
the dyes are, or how their blinking is distributed, enters the answer.  (It
would if the species had *different* brightness distributions; that
information is real, and deliberately not used, because it would assign
colours by brightness.)

So, with pi_c the fraction of the sample that is species c:

    P(c | I1, I2) = pi_c Binomial(I1 | N, p_c) / sum_j pi_j Binomial(I1 | N, p_j)

and the binomial coefficient C(N, I1) is common to every species too, leaving

    P(c | I1, I2) = pi_c p_c^I1 (1-p_c)^I2 / sum_j pi_j p_j^I1 (1-p_j)^I2.

That is the whole answer, in closed form, with no approximation.  For two
colours it is a logistic function of the counts, and the log-odds are
**linear** in them:

    log [ P(1 | I1, I2) / P(2 | I1, I2) ]
        = I1 log(p_1/p_2) + I2 log((1-p_1)/(1-p_2)) + log(pi_1/pi_2).

Three things follow.  The decision boundary is a straight line in the (I1, I2)
plane -- through the origin when the priors are equal, which is why the regions
in the intensity figure are wedges.  The evidence accumulates *per photon*:
each photon in channel 1 adds log(p_1/p_2) to the log-odds, so a molecule twice
as bright is twice as certain, on a log scale.  And the certainty is a property
of the counts, not of r: two molecules with the same ratio and different
brightness get different probabilities, which is the whole difference from
cutting the histogram.

A species whose own splitting ratio wanders -- across the field, between
molecules -- is the same derivation with p_c drawn from a Beta of mean p_c
rather than fixed.  The split is then beta-binomial, still closed-form, still
independent of the brightness:

    P(I1, I2 | c) ~ B(I1 + a_c, I2 + b_c) / B(a_c, b_c),
    a_c = p_c kappa_c,  b_c = (1 - p_c) kappa_c,
    kappa_c = (1 - rho_c^2) / spread^2 - 1,

where `spread` is that wandering as a standard deviation in r and kappa is the
concentration that reproduces it (var(p) = p(1-p)/(kappa+1), var(r) = 4 var(p)).
As spread -> 0, kappa -> infinity and it becomes the binomial again.

### Why not the Gaussian

The Gaussian in r is the normal approximation to this: substituting
I1 = N(1+r)/2 and Np_c = N(1+rho_c)/2 into the normal approximation of the
binomial gives back exactly

    (r - rho_c)^2 / (2 s_c^2),   s_c^2 = (1 - rho_c^2)/N,

so the two agree whenever both channels collect many photons -- the log-odds
are then quadratic in r instead of linear in the counts, and the difference is
second order.  They part company when one channel collects few, which is
precisely the ratiometric case worth having.  For two dyes splitting
2%/98% and 17%/83% (DECODE-Plex's validation pair, rho = -0.654 and +0.962) the
exact boundary is at r = +0.370 and the Gaussian's at r = +0.533, and
simulating from the true model at 8 detected photons:

| model | really misassigned | model claims |
|---|---|---|
| exact binomial | 0.303% | 0.310% |
| Gaussian in r  | 0.575% | 0.169% |

The Gaussian misassigns twice as many *and* under-states its own error by a
factor of three; the exact form is calibrated to the third digit.  For
symmetric ratios the two are indistinguishable.  Since the crosstalk budget is
only worth stating if the model can be believed, the implementation uses the
exact form -- `log_likelihoods` -- and keeps the Gaussian only as the variance
that the sigma-tolerance is measured in.

## Mode 2: probabilistic assignment with a crosstalk budget

With the modes rho_k as the hypotheses and pi_k as the prior fraction of each
species -- estimated from the counts between the minima, which is the one
number the histogram gives away for free -- the posterior for a localization is
an ordinary Gaussian mixture responsibility:

    L_k = exp( -(r - rho_k)^2 / (2 s_k^2) ) / s_k
    P(k | r, N) = pi_k L_k / sum_j pi_j L_j.

Assign to the best k only if it is good enough:

    k* = argmax_k P(k | r, N)
    assign k* if P(k* | r, N) >= 1 - c, otherwise 0.

That is Chow's rule -- a Bayes classifier with a reject option -- and the
reason for writing the threshold as 1 - c is that c is then exactly the
quantity the user wants to control.  The probability that an assigned
localization is wrong is 1 - P(k*|.), so

    E[ wrong | assigned ] = E[ 1 - P(k*|.) | assigned ] <= c.

**The allowed crosstalk is an upper bound on the expected fraction of
misassigned localizations among the assigned ones**, under this noise model.
It is not the fraction thrown away, and it is an upper bound, not an estimate:
the achieved crosstalk is the mean of 1 - P(k*|.) over the assigned
localizations, which is normally several times smaller than c because most
localizations are nowhere near the boundary.  Both numbers are reported, and
both are model-based -- they are as good as the claim that the modes are
Gaussian in r with the width the photon statistics give, which is why the
preview shows the fit against the histogram.

### How close is too close

A localization is left grey when the winning probability does not reach
``1 - crosstalk``: with c = 5%, anything below 0.95.  The threshold is not a
free knob, because the quantity it controls is the one that was asked for --
the chance that an assigned localization is wrong is 1 - P(winner), so
requiring P(winner) >= 1 - c bounds the expected error among the assigned by c.
It is also the optimal rule rather than a convenient one: under 0-1 loss with a
cost d for refusing, the Bayes-optimal decision is exactly to refuse when the
best posterior falls below 1 - d (Chow, 1970), so c is the exchange rate
between one wrong colour and one discarded localization.

Because the log-odds are linear in the counts, the band has an unusually
concrete width.  Each photon carries |log(p_1/p_2)| nats of evidence -- 1.386
for a 20/80 splitter -- and the rule needs ln((1-c)/c) nats, so the grey band
is a fixed number of *net photons*, whatever the brightness:

| crosstalk | evidence needed | net photons | band in r at N = 20 | at N = 1000 |
|---|---|---|---|---|
| 5%   | 2.94 nats | 2.1 | +-0.106 | +-0.0021 |
| 1%   | 4.60 nats | 3.3 | +-0.166 | +-0.0033 |
| 0.1% | 6.91 nats | 5.0 | +-0.249 | +-0.0050 |

Two photons of evidence buys 95% confidence, which is why the band is so narrow
in r for a bright localization and why tightening c does so little: the
requirement grows as ln(1/c), so a thousand-fold tighter budget asks for two
and a half times as many photons.  In the intensity plane it is a strip of
constant width along the boundary, which is what makes it a wedge in log-log.

This is also why the band alone is not enough -- it is about two photons wide,
and a localization that is no colour at all is nowhere near it.

### The posterior is relative, and that is not enough

A posterior divides the evidence between the species it was given.  It cannot
say *none of them*, and with well separated modes it barely ever rejects
anything: the ambiguous band is

    dr_eff = (1 - r^2) ln((1-c)/c) / (N_eff |rho_1 - rho_2|)

wide, which for two dyes 1.2 apart in r at 2000 effective photons and c = 5% is
0.001 -- a thousandth of the axis.  Worse, the only handle the budget gives is
logarithmic: going from c = 5% to c = 0.01% widens that band by a factor of
three, from nothing to nothing.  So a localization sitting in the valley
between the modes, 20 sigma from either species, is assigned with a posterior
of 0.999: a hair off the midpoint is enough to make one species overwhelmingly
likelier *than the other*, which is the only question asked.

Something has to ask the absolute question.  Each species' expected ratio rho_k
predicts the split of the photons that were actually detected, so the
observation can be tested against it:

    z_k = |r - rho_k| / s_k,     assign only if z_k* <= tolerance

with the same s_k as above -- shot noise for the photons this localization has,
plus `spread`.  Under H_k, z_k is a standard normal, so the tolerance is read in
sigma: 3 loses 0.3% of genuine localizations per species and refuses anything
the species could not have produced.  A localization that is far from every
species is not a hard call between two colours, it is a molecule that is
neither: two dyes at once, two emitters in one ROI, a fit that failed.  The two
tests refuse different things and are reported apart.

`tolerance = 0` turns it off and recovers the pure posterior, which is what
DECODE-Plex's rejection does.

### A population in the valley is not a boundary problem

The absolute test also covers the case that breaks *both* methods: a third
population that is not one of the species asked for.  In the histogram of r it
is a bump between two modes, with an empty stretch on each side of it -- so
"the lowest point between two modes" has two equally good answers, and whichever
is taken swallows that population whole.  Nothing about a boundary can fix
that, because the model is wrong rather than imprecise, so the summary says so
and names the bump: ask for another colour, or let the consistency test refuse
it.

### Two views of one decision

For two species the likelihood-ratio test reduces to a threshold on r, and it is worth
writing out because it shows what the mode buys over mode 1.  With equal priors
and s_1 = s_2 = s, the condition is

    |r - r_mid| >= s^2 ln((1-c)/c) / |rho_1 - rho_2|,     r_mid = (rho_1 + rho_2)/2

-- the same exclusion zone as mode 1, except that its half-width

    dr_eff = (1 - r^2) ln((1-c)/c) / (N_eff |rho_1 - rho_2|)

shrinks as 1/N_eff.  A bright localization is assigned right up to the
boundary; a dim one is rejected from a wide band around it; a localization that
is bright *and* far out is assigned with the confidence it has earned.  Mode 1
is the special case where every localization is treated as if it had the same
brightness, and dr has to be chosen for the worst of them.

The histogram of r divides the brightness out, and the brightness is what the
decision depends on, so there is a second figure: the two channels' counts
against each other, log-log, with the decided regions drawn over a 2D histogram
of the data.  A species is a *ray* from the origin there -- one line per
splitting ratio, a straight line of slope 1 in log-log -- and what the noise
model claims becomes a shape: a band that pinches in towards the ray as the
counts grow and flares out towards the origin where they are few.  `dr` draws
the same picture with the band edges parallel to the ray at a fixed distance,
and the two together are the argument for the probabilistic method in one look.
The regions are evaluated over a grid of intensities rather than sampled from
the data, which needs a stand-in for the fitted errors a grid point has not
got: the table's median N_eff/N, printed in the title.

## Where this comes from, and where it differs

DECODE-Plex (Methods, *Color Assignment*) does the same thing for C channels:
it normalises the predicted photon counts to a ratio vector, evaluates a
Gaussian likelihood of that vector under each colour's expected ratio with a
covariance built from the photon-count uncertainties divided by the total, and
normalises the likelihoods into a posterior.  With two channels, its ratio
vector's first component is p = N1/N and r = 2p - 1, so this is the same
construction on a stretched axis; the differences are all in the details that
two channels make visible.

* The covariance here is propagated through the *ratio*, not taken as the
  photon errors over N^2.  The denominator carries the same noise as the
  numerator, which is why the correct variance has both counts in it,
  4 (N2^2 s1^2 + N1^2 s2^2) / N^4, and why it collapses to the exact binomial
  (1 - r^2)/N rather than to something merely proportional to it.
* It is evaluated under the hypothesis, through N_eff, rather than at the
  measurement.  A localization at r = 0.98 has almost no room left for noise;
  a plug-in variance would read that as certainty about a species sitting at
  0.6.
* The expected ratios can be typed in -- measured on single-label samples, or
  computed from the dyes' spectra and the splitter, which is what DECODE-Plex
  assumes -- or read off the histogram's maxima, which is what a user with a
  mixed sample and no such measurement has.
* The rejection threshold is stated as a crosstalk budget rather than a
  posterior cut, because 1 - c is the same number with a meaning attached.
* The rejection is two tests, not one: the posterior threshold DECODE-Plex's
  rejection amounts to, and an absolute consistency test against each species'
  expected ratio.  Without the second, a localization that is no colour at all
  is assigned to the nearer one (see above), which matters more for a
  histogram-derived rho than for one measured on a single-label sample.
* `spread` exists because the Gaussian-in-r model is the part most likely to be
  wrong: real modes are broader than photon statistics.  The summary prints the
  strongest mode's measured width next to the shot-noise width, so the model
  can be checked against the histogram it claims to describe -- and when the
  mode is wider, the promised crosstalk is optimistic until `spread` accounts
  for the difference.

## Which parameter belongs to which method

Nothing is shared between the two: `dr` is the minima method's whole
mechanism and the probabilistic one never reads it, while `crosstalk`,
`consistency` and `extra spread` are the probabilistic model and the minima
method never reads them.  The GUI greys out whichever set is not in play,
because a number that does nothing should not look like a number that does.
Greying is presentation only -- the value stays, and a script sees every field.

## What is written

* `channel` -- int32, 0 for unassigned and 1..n for the species, left to right
  in r.  The name is not new: SMAP files carry it, `group` refuses to link
  across it (`GroupSettings.block_fields`), and the render tab already has a
  quick filter for it, so one column makes the result filterable, renderable
  and safe to group.
* `color_ratio` -- r itself, so it can be filtered and inspected.
* `channel_p1` ... `channel_pk` -- P(colour k | this molecule's counts), the
  answer itself.  `channel` is the decision that follows from it and cannot
  replace it: a molecule at 0.55 / 0.45 and one at 0.999 / 0.001 both read
  "colour 1".
* `channel_p` -- the posterior of the assigned species in mode 2, and 1 for an
  assigned localization in mode 1, where the decision carries no probability.
  0 wherever `channel` is 0.  Worth knowing what it is not: a *relative*
  probability, conditional on the localization being one of the species, so it
  sits at 0 or 1 everywhere except within a sliver of the boundary.
* `channel_sigma` -- how far the nearest species is, in its own sigma, for
  every localization whether assigned or not.  This is the one to filter on:
  it is what the consistency test cuts, and unlike `channel_p` it distinguishes
  a localization in a mode from one in the valley.

The modes are estimated from the current selection -- the filter, the ROI --
and applied to the whole table, as drift correction is.  A colour is a property
of the molecule, and there is no sense in which a localization outside the
current ROI has a different one.
