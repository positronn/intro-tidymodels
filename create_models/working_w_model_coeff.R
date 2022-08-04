# create_models/working_w_model_coeff.R
#
#
library(tidymodels)
tidymodels_prefer()
theme_set(theme_bw())

data(Chicago)

Chicago <- Chicago %>% select(ridership, Clark_Lake, Austin, Harlem)

# Let’s start by fitting only a single parsnip model object. We’ll create a model specification using linear_reg().
# The fit() function estimates the model coefficients, given a formula and data set.
lm_spec <- linear_reg()
lm_fit <- fit(lm_spec, ridership ~ ., data = Chicago)
lm_fit

# The best way to retrieve the fitted parameters is to use the tidy() method.
# This function, in the broom package, returns the coefficients and their associated statistics
# in a data frame with standardized column names:
tidy(lm_fit)


# We’ll use five bootstrap resamples of the data to simplify the plots and output
# (normally, we would use a larger number of resamples for more reliable estimates).
set.seed(123)
bt <- bootstraps(Chicago, times = 5)

# With resampling, we fit the same model to the different simulated versions of the data
# set produced by resampling. The tidymodels function fit_resamples() is the recommended
# approach for doing so.

get_lm_coefs <- function(model) {
    model %>% 
        # get the lm model object
        extract_fit_engine() %>% 
        # transform its format
        tidy()
}
tidy_ctrl <- control_grid(extract = get_lm_coefs)

# This argument is then passed to fit_resamples():
lm_res <- 
    lm_spec %>% 
    fit_resamples(ridership ~ ., resamples = bt, control = tidy_ctrl)
lm_res

# Note that there is a .extracts column in our resampling results. This object contains
# the output of our get_lm_coefs() function for each resample. The structure of the elements
# of this column is a little complex. Let’s start by looking at the first element 
# (which corresponds to the first resample):
lm_res$.extracts[[1]]

# There is another column in this element called .extracts that has the results of the tidy() function call:
lm_res$.extracts[[1]]$.extracts[[1]]

# These nested columns can be flattened via the tidyr unnest() function:
lm_res %>% 
    select(id, .extracts) %>% 
    unnest(.extracts) 

# We still have a column of nested tibbles, so we can run the same command again
# to get the data into a more useful format:
lm_coefs <- 
    lm_res %>% 
    select(id, .extracts) %>% 
    unnest(.extracts) %>% 
    unnest(.extracts)

lm_coefs %>%
    select(id, term, estimate, p.value)


# That’s better! Now, let’s plot the model coefficients for each resample:
lm_coefs %>%
    filter(term != "(Intercept)") %>% 
    ggplot(aes(x = term, y = estimate, group = id, col = id)) +  
    geom_hline(yintercept = 0, lty = 3) + 
    geom_line(alpha = 0.3, lwd = 1.2) + 
    labs(y = "Coefficient", x = NULL) +
    theme(legend.position = "top")

# There seems to be a lot of uncertainty in the coefficient for the Austin station data,
# but less for the other two.

# Looking at the code for unnesting the results, you may find the double-nesting
# structure excessive or cumbersome. However, the extraction functionality is flexible,
# and a simpler structure would prevent many use cases.


glmnet_spec <- 
    linear_reg(penalty = 0.1, mixture = 0.95) %>% 
    set_engine("glmnet")

glmnet_wflow <- 
    workflow() %>% 
    add_model(glmnet_spec) %>% 
    add_formula(ridership ~ .)

glmnet_fit <-
    fit(glmnet_wflow, Chicago)
glmnet_fit

# In this output, the term lambda is used to represent the penalty.

# Using glmnet penalty values
# This glmnet fit contains multiple penalty values which depend on the data set;
# changing the data (or the mixture amount) often produces a different set of values. 
# For this data set, there are 55 penalties available. To get the set of penalties
# produced for this data set, we can extract the engine fit and tidy:
glmnet_fit %>% 
    extract_fit_engine() %>% 
    tidy() %>% 
    rename(penalty = lambda) %>%   # <- for consistent naming
    filter(term != "(Intercept)")

# This works well but, it turns out that our penalty value (0.1) is not in the list
# produced by the model! The underlying package has functions that use interpolation to
# produce coefficients for this specific value, but the tidy() method for glmnet objects
# does not use it.

# Using specific penalty values
# If we run the tidy() method on the workflow or parsnip object, a different
# function is used that returns the coefficients for the penalty value that we specified:
tidy(glmnet_fit)

# For any another (single) penalty, we can use an additional argument:
tidy(glmnet_fit, penalty = 5.5620)  # A value from above

# The reason for having two tidy() methods is that, with tidymodels, the focus is on using a specific penalty value.

# Let’s tune our glmnet model over both parameters with this grid:
pen_vals <- 10^seq(-3, 0, length.out = 10)
grid <- crossing(penalty = pen_vals, mixture = c(0.1, 1.0))

glmnet_tune_spec <- 
    linear_reg(penalty = tune(),
               mixture = tune()) %>% 
    set_engine("glmnet",
               path_values = pen_vals)

glmnet_wflow <- 
    glmnet_wflow %>% 
    update_model(glmnet_tune_spec)

# Now we will use an extraction function similar to when we used ordinary least squares.
# We add an additional argument to retain coefficients that are shrunk to zero by the lasso penalty:
get_glmnet_coefs <- function(x) {
    x %>% 
        extract_fit_engine() %>% 
        tidy(return_zeros = TRUE) %>% 
        rename(penalty = lambda)
}
parsnip_ctrl <- control_grid(extract = get_glmnet_coefs)

glmnet_res <- 
    glmnet_wflow %>% 
    tune_grid(
        resamples = bt,
        grid = grid,
        control = parsnip_ctrl
    )
glmnet_res

# As noted before, the elements of the main .extracts column have an embedded list
# column with the results of get_glmnet_coefs():
glmnet_res$.extracts[[1]] %>% head()
glmnet_res$.extracts[[1]]$.extracts[[1]] %>% head()

# As before, we’ll have to use a double unnest(). Since the penalty value is in
# both the top-level and lower-level .extracts, we’ll use select() to get rid of the first
# version (but keep mixture):
glmnet_res %>% 
    select(id, .extracts) %>% 
    unnest(.extracts) %>% 
    select(id, mixture, .extracts) %>%  # <- removes the first penalty column
    unnest(.extracts)

# But wait! We know that each glmnet fit contains all of the coefficients.
# This means, for a specific resample and value of mixture, the results are the same:



