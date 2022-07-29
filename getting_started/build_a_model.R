# getting_started/build_a_model.R
#
#

# We start with data for modeling, learn how to specify and train models
# with different engines using the parsnip package, and understand why these
# functions are designed this way.

# for the parsnip package, along with the rest of tidymodels
library(tidymodels)  

# Helper packages
library(readr)       # for importing data
library(broom.mixed) # for converting bayesian models to tidy tibbles
library(dotwhisker) 



# Let’s use the data from Constable (1993) to explore how three different
# feeding regimes affect the size of sea urchins over time. The initial size of
# the sea urchins at the beginning of the experiment probably affects how big
# they grow as they are fed.
urchins <-
    # Data were assembled for a tutorial 
    # at https://www.flutterbys.com.au/stats/tut/tut7.5a.html
    read_csv('data/urchins.csv') %>% 
    # Change the names to be a little more verbose
    setNames(c("food_regime", "initial_volume", "width")) %>% 
    # Factors are very helpful for modeling, so we convert one column
    mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))


#   Let’s take a quick look at the data:
#   `food_regime`:
#       experimental feeding regime group
#           either: `Initial`, `Low` or `High`
#   `initial_volume`:
#       size in millimiters at the start of the experiment
#   `width`:
#       suture wifth at the end of the experiments
urchins


# As a first step in modeling, it’s always a good idea to plot the data:
ggplot(urchins,
       aes(x = initial_volume, 
           y = width, 
           group = food_regime, 
           col = food_regime)) + 
    geom_point() + 
    geom_smooth(method = lm, se = FALSE, formula = 'y ~ x') +
    scale_color_viridis_d(option = "plasma", end = .7)

# We can see that urchins that were larger in volume at the start of
# the experiment tended to have wider sutures at the end, but the slopes
# of the lines look different so this effect may depend on the feeding regime condition.


#                         * Build and fit a model *
# 
# A standard two-way analysis of variance (ANOVA) model makes sense
# for this dataset because we have both a continuous predictor and a categorical predictor.
# Since the slopes appear to be different for at least two of the feeding regimes,
# let’s build a model that allows for two-way interactions.
#           width ~ initial_volume * food_regime
# allows our regression model depending on initial volume to have separate slopes
# and intercepts for each food regime.

# For this kind of model, ordinary least squares is a good initial approach.
# With tidymodels, we start by specifying the functional form of the model that we
# want using the parsnip package. Since there is a numeric outcome and the model should be
# linear with slopes and intercepts, the model type is “linear regression”. We can declare this with:
linear_reg()
lm_mod <- linear_reg()

# From here, the model can be estimated or trained using the fit() function:
lm_fit <-
    lm_mod %>% 
    fit(width ~ initial_volume * food_regime, data = urchins)
lm_fit

# Perhaps our analysis requires a description of the model parameter estimates and
# their statistical properties. Although the summary() function for lm objects can provide that,
# it gives the results back in an unwieldy format. Many models have a tidy() method that
# provides the summary results in a more predictable and useful format (e.g. a data
# frame with standard column names):
tidy(lm_fit)

# This kind of output can be used to generate a dot-and-whisker plot of our
# regression results using the dotwhisker package:


tidy(lm_fit) %>% 
    dwplot(dot_args = list(size = 2, color = "black"),
           whisker_args = list(color = "black"),
           vline = geom_vline(xintercept = 0, colour = "grey50", linetype = 2))


#                           * Use a model to predict *
#
# This fitted object lm_fit has the lm model output built-in, which you can
# access with lm_fit$fit, but there are some benefits to using the fitted parsnip
# model object when it comes to predicting. 
# Suppose that, for a publication, it would be particularly interesting to make a plot
# of the mean body size for urchins that started the experiment with an initial
# volume of 20ml. To create such a graph, we start with some new example data that we
# will make predictions for, to show in our graph:
new_points <- expand.grid(initial_volume = 20, 
                          food_regime = c("Initial", "Low", "High"))
new_points

# To get our predicted results, we can use the predict() function to find the mean values at 20ml.
# It is also important to communicate the variability, so we also need to find the predicted
# confidence intervals. If we had used lm() to fit the model directly, a few minutes
# of reading the documentation page for predict.lm() would explain how to do this. However,
# if we decide to use a different model to estimate urchin size (spoiler: we will!),
# it is likely that a completely different syntax would be required.

# Instead, with tidymodels, the types of predicted values are standardized so that we
# can use the same syntax to get these values.

# First, let’s generate the mean body width values:
mean_pred <- predict(lm_fit, new_data = new_points)
mean_pred

# When making predictions, the tidymodels convention is to always produce a
# tibble of results with standardized column names. This makes it easy to combine
# the original data and the predictions in a usable format:
conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")
conf_int_pred

# Now combine: 
plot_data <- 
    new_points %>% 
    bind_cols(mean_pred) %>% 
    bind_cols(conf_int_pred)

# and plot:
ggplot(plot_data, aes(x = food_regime)) + 
    geom_point(aes(y = .pred)) + 
    geom_errorbar(aes(ymin = .pred_lower, 
                      ymax = .pred_upper),
                  width = .2) + 
    labs(y = "urchin size")


#                           * Model with a different engine *
# 
# Every one on your team is happy with that plot except that one person who just
# read their first book on Bayesian analysis. They are interested in knowing if the
# results would be different if the model were estimated using a Bayesian approach. In
# such an analysis, a prior distribution needs to be declared for each model parameter that
# represents the possible values of the parameters (before being exposed to the observed data).
# After some discussion, the group agrees that the priors should be bell-shaped but,
# since no one has any idea what the range of values should be, to take a conservative
# approach and make the priors wide using a Cauchy distribution (which is the same as a
# t-distribution with a single degree of freedom).

# The documentation on the rstanarm package shows us that the stan_glm() function
# can be used to estimate this model, and that the function arguments that need to be
# specified are called prior and prior_intercept. It turns out that linear_reg() has a stan engine.
# Since these prior distribution arguments are specific to the Stan software,
# they are passed as arguments to parsnip::set_engine().
# After that, the same exact fit() call is used:
# set the prior distribution
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# make the parsnip model
bayes_mod <-   
    linear_reg() %>% 
    set_engine("stan", 
               prior_intercept = prior_dist, 
               prior = prior_dist) 

# train the model
bayes_fit <- 
    bayes_mod %>% 
    fit(width ~ initial_volume * food_regime, data = urchins)

print(bayes_fit, digits = 5)

# To update the parameter table, the tidy() method is once again used:
tidy(bayes_fit, conf.int = TRUE)

# A goal of the tidymodels packages is that the interfaces to common tasks are
# standardized (as seen in the tidy() results above). The same is true
# for getting predictions; we can use the same code even though the underlying
# packages use very different syntax:
bayes_plot_data <- 
    new_points %>% 
    bind_cols(predict(bayes_fit, new_data = new_points)) %>%                # pred for the mean
    bind_cols(predict(bayes_fit, new_data = new_points, type = "conf_int")) # pred for conf int

ggplot(bayes_plot_data, aes(x = food_regime)) + 
    geom_point(aes(y = .pred)) + 
    geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
    labs(y = "urchin size") + 
    ggtitle("Bayesian model with t(1) prior distribution")

# This isn’t very different from the non-Bayesian results (except in interpretation).


#                       * Why does it work that way? *
# The extra step of defining the model using a function like linear_reg() might seem superfluous
# since a call to lm() is much more succinct. However, the problem with standard modeling
# functions is that they don’t separate what you want to do from the execution. For example,
# the process of executing a formula has to happen repeatedly across model calls even when the
# formula does not change; we can’t recycle those computations.

# Also, using the tidymodels framework, we can do some interesting things by incrementally
# creating a model (instead of using single function call). Model tuning with tidymodels uses
# the specification of the model to declare what parts of the model should be tuned. That would
# be very difficult to do if linear_reg() immediately fit the model.

# If you are familiar with the tidyverse, you may have noticed that our modeling code
# uses the magrittr pipe (%>%). With dplyr and other tidyverse packages, the pipe works
# well because all of the functions take the data as the first argument. For example:
urchins %>% 
    group_by(food_regime) %>% 
    summarize(med_vol = median(initial_volume))

# whereas the modeling code uses the pipe to pass around the model object:
bayes_mod %>% 
    fit(width ~ initial_volume * food_regime, data = urchins)

# This may seem jarring if you have used dplyr a lot, but it is extremely similar
# to how ggplot2 operates:
ggplot(urchins,
       aes(initial_volume, width)) +      # returns a ggplot object 
    geom_jitter() +                         # same
    geom_smooth(method = lm, se = FALSE, formula = 'y ~ x') +  # same                    
    labs(x = "Volume", y = "Width")         # etc
