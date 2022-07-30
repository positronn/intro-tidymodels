# statistical_analysis/bootstrap_resampling.R
# Bootstrap resampling and tidy regression models
# Apply bootstrap resampling to estimate uncertainty in model parameters.

# Bootstrapping consists of randomly sampling a data set with replacement,
# then performing the analysis individually on each bootstrapped replicate.
# The variation in the resulting estimate is then a reasonable approximation
# of the variance in our estimate.

library(tidymodels)

# Let’s say we want to fit a nonlinear model to the weight/mileage
# relationship in the mtcars data set.
ggplot(mtcars, aes(mpg, wt)) + 
    geom_point()

# We might use the method of nonlinear least squares (via the nls() function)
# to fit a model.
nlsfit <-
    nls(
        mpg ~ k / wt + b,
        mtcars,
        start = list(k = 1, b = 0)
    )
summary(nlsfit)


ggplot(mtcars, aes(wt, mpg)) +
    geom_point() +
    geom_line(aes(y = predict(nlsfit)))

# While this does provide a p-value and confidence intervals for
# the parameters, these are based on model assumptions that may not
# hold in real data. Bootstrapping is a popular method for providing
# confidence intervals and predictions that are more robust to the nature of the data.

#                           * Bootstrapping models *
#
# We can use the bootstraps() function in the rsample package to sample
# bootstrap replications. First, we construct 2000 bootstrap replicates
# of the data, each of which has been randomly sampled with replacement.
# The resulting object is an rset, which is a data frame with a column of rsplit objects.

# An rsplit object has two main components: an analysis data set and an
# assessment data set, accessible via analysis(rsplit) and assessment(rsplit)
# respectively. For bootstrap samples, the analysis data set is the bootstrap
# sample itself, and the assessment data set consists of all the out-of-bag samples.
set.seed(27)
boots <- bootstraps(mtcars, times = 2000, apparent = TRUE)
boots

# Let’s create a helper function to fit an nls() model on each bootstrap
# sample, and then use purrr::map() to apply this function to all the bootstrap
# samples at once. Similarly, we create a column of tidy coefficient information
# by unnesting.
fit_nls_on_bootstrap <- function(split) {
    nls(mpg ~ k / wt + b, analysis(split), start = list(k = 1, b = 0))
}

boot_models <-
    boots %>% 
    mutate(model = map(splits, fit_nls_on_bootstrap),
           coef_info = map(model, tidy))

boot_coefs <- 
    boot_models %>% 
    unnest(coef_info)
boot_coefs


#                           * Confidence intervals *
#
# We can then calculate confidence intervals (using what is called the percentile method):
percentile_intervals <-
    int_pctl(boot_models, coef_info)
percentile_intervals
# Or we can use histograms to get a more detailed idea of the uncertainty in each estimate:
ggplot(boot_coefs, aes(estimate)) +
    geom_histogram(bins = 30,
                   alpha = 0.6,
                   fill = 'steelblue') +
    facet_wrap( ~ term, scales = "free") +
    geom_vline(aes(xintercept = .lower), data = percentile_intervals, col = "blue") +
    geom_vline(aes(xintercept = .upper), data = percentile_intervals, col = "blue")

# The rsample package also has functions for other types of confidence intervals.

#                           * Possible model fits *
# 
# We can use augment() to visualize the uncertainty in the fitted curve.
# Since there are so many bootstrap samples, we’ll only show a sample of
# the model fits in our visualization:
boot_aug <- 
    boot_models %>% 
    sample_n(150) %>% 
    mutate(augmented = map(model, augment)) %>% 
    unnest(augmented)

boot_aug

ggplot(boot_aug, aes(wt, mpg)) +
    geom_line(aes(y = .fitted, group = id), alpha = .2, col = "blue") +
    geom_point()

# With only a few small changes, we could easily perform bootstrapping with
# other kinds of predictive or hypothesis testing models, since the tidy()
# and augment() functions works for many statistical outputs. As another example,
# we could use smooth.spline(), which fits a cubic smoothing spline to data:
fit_spline_on_bootstrap <- function(split) {
    data <- analysis(split)
    smooth.spline(data$wt, data$mpg, df = 4)
}

boot_splines <- 
    boots %>% 
    sample_n(200) %>% 
    mutate(spline = map(splits, fit_spline_on_bootstrap),
           aug_train = map(spline, augment))

splines_aug <- 
    boot_splines %>% 
    unnest(aug_train)

ggplot(splines_aug, aes(x, y)) +
    geom_line(aes(y = .fitted, group = id), alpha = 0.2, col = "blue") +
    geom_point()

