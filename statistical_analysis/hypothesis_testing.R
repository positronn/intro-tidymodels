# statistical_analysis/hypothesis_testing.R
# Hypothesis testing using resampling and tidy data
# 
# The tidymodels package infer implements an expressive grammar to
# perform statistical inference that coheres with the tidyverse design framework.
# Rather than providing methods for specific statistical tests, this package
# consolidates the principles that are shared among common hypothesis tests
# into a set of 4 main verbs (functions), supplemented with many utilities to
# visualize and extract information from their outputs.
# Regardless of which hypothesis test we’re using, we’re still asking the same kind of question:
#       Is the effect or difference in our observed data real, or due to chance?
#
# To answer this question, we start by assuming that the observed data came
# from some world where “nothing is going on” (i.e. the observed effect was simply
# due to random chance), and call this assumption our null hypothesis. (In reality,
# we might not believe in the null hypothesis at all; the null hypothesis is in opposition
# to the alternate hypothesis, which supposes that the effect present in the observed data
# is actually due to the fact that “something is going on.”) We then calculate a test
# statistic from our data that describes the observed effect. We can use this test statistic
# to calculate a p-value, giving the probability that our observed data could come about
# if the null hypothesis was true. If this probability is below some pre-defined significance
# level alpha, then we can reject our null hypothesis.

# specify() allows you to specify the variable, or relationship between variables,
# that you’re interested in,
# hypothesize() allows you to declare the null hypothesis,
# generate() allows you to generate data reflecting the null hypothesis, and
# calculate() allows you to calculate a distribution of statistics from the
#   generated data to form the null distribution.

library(tidymodels) # Includes the infer package

data(gss)

# take a look at gss structure
dplyr::glimpse(gss)

# Each row is an individual survey response, containing some basic
# demographic information on the respondent as well as some additional variables.
# See ?gss for more information on the variables included and their source. Note that
# this data (and our examples on it) are for demonstration purposes only, and will
# not necessarily provide accurate estimates unless weighted properly. For these examples,
# let’s suppose that this data set is a representative sample of a population we
# want to learn about: American adults.

#                           * Specify variables *
#
# The specify() function can be used to specify which of the variables in the data
# set you’re interested in. If you’re only interested in, say, the age of the respondents,
# you might write:
gss %>%
    specify(response = age)

# On the front end, the output of specify() just looks like it selects off
# the columns in the dataframe that you’ve specified. What do we see if we check the
# class of this object, though?
gss %>%
    specify(response = age) %>%
    class()
# We can see that the infer class has been appended on top of the dataframe classes;
# this new class stores some extra metadata.
# If you’re interested in two variables (age and partyid, for example) you can specify()
# their relationship in one of two (equivalent) ways:
# as a formula
gss %>%
    specify(age ~ partyid)


# with the named arguments
gss %>%
    specify(response = age, explanatory = partyid)

# If you’re doing inference on one proportion or a difference in proportions,
# you will need to use the success argument to specify which level of your
# response variable is a success. For instance, if you’re interested in the proportion
# of the population with a college degree, you might use the following code:
# specifying for inference on proportions
gss %>%
    specify(response = college, success = "degree")


#                               * Declare the hypothesis *
#
# The next step in the infer pipeline is often to declare a null hypothesis using hypothesize().
# The first step is to supply one of “independence” or “point” to the null argument.
# If your null hypothesis assumes independence between two variables, then this is all
# you need to supply to hypothesize():
gss %>%
    specify(college ~ partyid, success = "degree") %>%
    hypothesize(null = "independence")
# If you’re doing inference on a point estimate, you will also need to provide one
# of p (the true proportion of successes, between 0 and 1), mu (the true mean),
# med (the true median), or sigma (the true standard deviation). For instance,
# if the null hypothesis is that the mean number of hours worked per week in our
# population is 40, we would write:
gss %>%
    specify(response = hours) %>%
    hypothesize(null = "point", mu = 40)
# Again, from the front-end, the dataframe outputted from hypothesize() looks
# almost exactly the same as it did when it came out of specify(), but infer
# now “knows” your null hypothesis.

#                                   * Generate the distribution *
#
# Once we’ve asserted our null hypothesis using hypothesize(), we can construct a null
# distribution based on this hypothesis. We can do this using one of several methods,
# supplied in the type argument:
# bootstrap: A bootstrap sample will be drawn for each replicate, where a
#   sample of size equal to the input sample size is drawn (with replacement) from the
#   input sample data.
# permute: For each replicate, each input value will be randomly reassigned
#   (without replacement) to a new output value in the sample.
# simulate: A value will be sampled from a theoretical distribution with parameters
#   specified in hypothesize() for each replicate. (This option is currently only
#   applicable for testing point estimates.)

# Continuing on with our example above, about the average number of hours worked a week, we might write:
gss %>%
    specify(response = hours) %>%
    hypothesize(null = "point", mu = 40) %>%
    generate(reps = 5000, type = "bootstrap")

# In the above example, we take 5000 bootstrap samples to form our null distribution.
# To generate a null distribution for the independence of two variables, we could also
# randomly reshuffle the pairings of explanatory and response variables to break any
# existing association. For instance, to generate 5000 replicates that can be used to
# create a null distribution under the assumption that political party affiliation is not
# affected by age:
gss %>%
    specify(partyid ~ age) %>%
    hypothesize(null = "independence") %>%
    generate(reps = 5000, type = "permute")

#                                   * Calculate statistics *
#
# Depending on whether you’re carrying out computation-based inference or
# theory-based inference, you will either supply calculate() with the output of
# generate() or hypothesize(), respectively. The function, for one, takes in a stat argument,
# which is currently one of "mean", "median", "sum", "sd", "prop", "count", "diff in means",
# "diff in medians", "diff in props", "Chisq", "F", "t", "z", "slope", or "correlation".
# For example, continuing our example above to calculate the null distribution of mean
# hours worked per week:
gss %>%
    specify(response = hours) %>%
    hypothesize(null = "point", mu = 40) %>%
    generate(reps = 5000, type = "bootstrap") %>%
    calculate(stat = "mean")

# The output of calculate() here shows us the sample statistic (in this case, the mean)
# for each of our 1000 replicates. If you’re carrying out inference on differences in means,
# medians, or proportions, or "t" and "z" statistics, you will need to supply an order
# argument, giving the order in which the explanatory variables should be subtracted.
# For instance, to find the difference in mean age of those that have a college degree and
# those that don’t, we might write:
gss %>%
    specify(age ~ college) %>%
    hypothesize(null = "independence") %>%
    generate(reps = 5000, type = "permute") %>%
    calculate("diff in means", order = c("degree", "no degree"))

#                               * Other utilities *
# 
# The infer package also offers several utilities to extract meaning out of summary
# statistics and null distributions; the package provides functions to visualize where
# a statistic is relative to a distribution (with visualize()),
# calculate p-values (with get_p_value()), and calculate confidence
# intervals (with get_confidence_interval()).

# To illustrate, we’ll go back to the example of determining whether the mean number
# of hours worked per week is 40 hours.
# find the point estimate
point_estimate <- gss %>%
    specify(response = hours) %>%
    calculate(stat = "mean")

# generate a null distribution
null_dist <- gss %>%
    specify(response = hours) %>%
    hypothesize(null = "point", mu = 40) %>%
    generate(reps = 5000, type = "bootstrap") %>%
    calculate(stat = "mean")

# (Notice the warning: Removed 1244 rows containing missing values. This would be worth noting
# if you were actually carrying out this hypothesis test.)
# Our point estimate 41.382 seems pretty close to 40, but a little bit different.
# We might wonder if this difference is just due to random chance, or if the mean
# number of hours worked per week in the population really isn’t 40.

# We could initially just visualize the null distribution.
null_dist %>%
    visualize(bins=30)

# Where does our sample’s observed statistic lie on this distribution?
# We can use the obs_stat argument to specify this.
null_dist %>%
    visualize(bins=15) +
    shade_p_value(obs_stat = point_estimate, direction = "two_sided", size=0.6 )

# Notice that infer has also shaded the regions of the null distribution that
# are as (or more) extreme than our observed statistic. (Also, note that we now
# use the + operator to apply the shade_p_value() function. This is because visualize()
# outputs a plot object from ggplot2 instead of a dataframe, and the + operator
# is needed to add the p-value layer to the plot object.) The red bar looks like it’s
# slightly far out on the right tail of the null distribution, so observing a sample
# mean of 41.382 hours would be somewhat unlikely if the mean was actually 40 hours.
# How unlikely, though?

# get a two-tailed p-value
p_value <- null_dist %>%
    get_p_value(obs_stat = point_estimate, direction = "two_sided")

p_value

# It looks like the p-value is 0.038, which is pretty small—if the true mean number
# of hours worked per week was actually 40, the probability of our sample mean being
# this far (1.382 hours) from 40 would be 0.038. This may or may not be statistically
# significantly different, depending on the significance level alpha you decided on
# before you ran this analysis. If you had set alpha = .05, then this difference would be
# statistically significant, but if you had set alpha = .01, then it would not be.

# To get a confidence interval around our estimate, we can write:
# start with the null distribution
null_dist %>%
    # calculate the confidence interval around the point estimate
    get_confidence_interval(point_estimate = point_estimate,
                            # at the 95% confidence level
                            level = .95,
                            # using the standard error
                            type = "se")

# As you can see, 40 hours per week is not contained in this interval,
# which aligns with our previous conclusion that this finding is significant at
# the confidence level alpha = .05.


#                           * Theoretical methods *
#
# The infer package also provides functionality to use theoretical methods for
# "Chisq", "F", "t" and "z" distributions.

# Generally, to find a null distribution using theory-based methods, use the
# same code that you would use to find the observed statistic elsewhere, replacing
# calls to calculate() with assume(). For example, to calculate the observed "t"
# statistic (a standardized mean):
# calculate an observed t statistic
obs_t <- gss %>%
    specify(response = hours) %>%
    hypothesize(null = "point", mu = 40) %>%
    calculate(stat = "t")

# switch out `calculate()` with `assume()` to define a distribution
t_dist <- gss %>%
    specify(response = hours) %>%
    assume(distribution = "t")

# visualize the theoretical null distribution
visualize(t_dist) +
    shade_p_value(obs_stat = obs_t, direction = "greater")

# more exactly, calculate the p-value
get_p_value(t_dist, obs_t, "greater")


# Confidence intervals lie on the scale of the data rather than on the
# standardized scale of the theoretical distribution, so be sure to use the
# unstandardized observed statistic when working with confidence intervals.
# calculate the point estimate
obs_mean <- gss %>%
    specify(response = hours) %>%
    calculate(stat = "mean")

# find the theory-based confidence interval
theor_ci <- 
    get_confidence_interval(
        x = t_dist,
        level = .95,
        point_estimate = obs_mean
    )

theor_ci

# When visualized, the "t" distribution will be recentered and rescaled to
# align with the scale of the observed data.
# visualize the theoretical sampling distribution
visualize(t_dist) +
    shade_confidence_interval(theor_ci)


#                               * Multiple regression *
#
# To accommodate randomization-based inference with multiple explanatory variables,
# the package implements an alternative workflow based on model fitting. Rather than
# calculate()ing statistics from resampled data, this side of the package allows
# you to fit() linear models on data resampled according to the null hypothesis,
# supplying model coefficients for each explanatory variable. For the most part,
# you can just switch out calculate() for fit() in your calculate()-based workflows.
# As an example, suppose that we want to fit hours worked per week using the respondent
# age and college completion status. We could first begin by fitting a linear model to the observed data.
observed_fit <- gss %>%
    specify(hours ~ age + college) %>%
    fit()

# Now, to generate null distributions for each of these terms, we can fit 1000
# models to resamples of the gss dataset, where the response hours is permuted in
# each. Note that this code is the same as the above except for the addition of
# the hypothesize() and generate() step.
null_fits <- gss %>%
    specify(hours ~ age + college) %>%
    hypothesize(null = "independence") %>%
    generate(reps = 1000, type = "permute") %>%
    fit()

null_fits

# To permute variables other than the response variable, the variables argument
# to generate() allows you to choose columns from the data to permute. Note that
# any derived effects that depend on these columns (e.g., interaction effects) will
# also be affected.
# Beyond this point, observed fits and distributions from null fits interface
# exactly like analogous outputs from calculate(). For instance, we can use the
# following code to calculate a 95% confidence interval from these objects.
get_confidence_interval(
    null_fits, 
    point_estimate = observed_fit, 
    level = .95
)

# Or, we can shade p-values for each of these observed regression coefficients from
# the observed data.
visualize(null_fits) + 
    shade_p_value(observed_fit, direction = "both", size=0.7)



