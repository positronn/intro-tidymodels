# statistical_analysis/corr_and_reg_fundamentals.R
# Correlation and regression fundamentals with tidy data principles
#
#
# While the tidymodels package broom is useful for summarizing the result of
# a single analysis in a consistent format, it is really designed for high-throughput
# applications, where you must combine results from multiple analyses. These could
# be subgroups of data, analyses using different models, bootstrap replicates,
# permutations, and so on. In particular, it plays well with the nest()/unnest()
# functions from tidyr and the map() function in purrr.
library(tidymodels)
library(ggplot2)

data(Orange)
Orange <- as_tibble(Orange)
Orange


# This contains 35 observations of three variables: Tree, age, and circumference.
# Tree is a factor with five levels describing five trees. As might be expected,
# age and circumference are correlated:
cor(Orange$age, Orange$circumference)

Orange %>% 
    ggplot(mapping = aes(x = age,
                         y = circumference,
                         color = Tree)) +
    geom_line()


# Suppose you want to test for correlations individually within each tree.
# You can do this with dplyr’s group_by
Orange %>% 
    group_by(Tree) %>%
    summarize(correlation = cor(age, circumference))

# (Note that the correlations are much higher than the aggregated one,
# and also we can now see the correlation is similar across trees).

# Suppose that instead of simply estimating a correlation, we want to perform a
# hypothesis test with cor.test():
ct <- cor.test(Orange$age, Orange$circumference)
ct

# This test output contains multiple values we may be interested in.
# Some are vectors of length 1, such as the p-value and the estimate,
# and some are longer, such as the confidence interval. We can get this into a
# nicely organized tibble using the tidy() function:
tidy(ct)


# Often, we want to perform multiple tests or fit multiple models, each on a
# different part of the data. In this case, we recommend a nest-map-unnest workflow.
# For example, suppose we want to perform correlation tests for each different tree.
# We start by nesting our data based on the group of interest:
nested <-
    Orange %>% 
    nest(data = c(age, circumference))

# Then we perform a correlation test for each nested tibble using purrr::map():
nested %>% 
    mutate(test = map(data, ~ cor.test(.x$age, .x$circumference)))

# This results in a list-column of S3 objects. We want to tidy each of the objects,
# which we can also do with map().
nested %>% 
    mutate(
        test = map(data, ~ cor.test(.x$age, .x$circumference)),
        tidied = map(test, tidy)
    )
# Finally, we want to unnest the tidied data frames so we can see the
# results in a flat tibble. All together, this looks like:
Orange %>% 
    nest(data = c(age, circumference)) %>% 
    mutate(
        test = map(data, ~ cor.test(.x$age, .x$circumference)), # S3 list-col
        tidied = map(test, tidy)
    ) %>% 
    unnest(cols = tidied) %>% 
    select(-data, -test)


#                       * Regression models *
#
# This type of workflow becomes even more useful when applied to regressions.
# Untidy output for a regression looks like:
lm_fit <- lm(age ~ circumference, data = Orange)
summary(lm_fit)

# When we tidy these results, we get multiple rows of output for each model:
tidy(lm_fit)

# Now we can handle multiple regressions at once using exactly the same workflow as before:
Orange %>% 
    nest(data = c(-Tree)) %>% 
    mutate(
        fit = map(data, ~ lm(age ~ circumference, data = .x)),
        tidied = map(fit, tidy)
    ) %>% 
    unnest(tidied) %>% 
    select(-data, -fit)

#You can just as easily use multiple predictors in the regressions,
# as shown here on the mtcars dataset. We nest the data into automatic vs.
# manual cars (the am column), then perform the regression within each nested tibble.    
data(mtcars)
mtcars <- as_tibble(mtcars)    
mtcars    

mtcars %>%
    nest(data = c(-am)) %>% 
    mutate(
        fit = map(data, ~ lm(wt ~ mpg + qsec + gear, data = .x)),  # S3 list-col
        tidied = map(fit, tidy)
    ) %>% 
    unnest(tidied) %>% 
    select(-data, -fit)    

# What if you want not just the tidy() output, but the augment() and glance()
# outputs as well, while still performing each regression only once? Since we’re using
# list-columns, we can just fit the model once and use multiple list-columns to
# store the tidied, glanced and augmented outputs.
regressions <-
    mtcars %>% 
    nest(data = c(-am)) %>% 
    mutate(
        fit = map(data, ~ lm(wt ~ mpg + qsec + gear, data = .x)),
        tidied = map(fit, tidy),
        glanced = map(fit, glance),
        augmented = map(fit, augment)
    )

regressions %>% 
    select(tidied) %>% 
    unnest(tidied)

regressions %>% 
    select(glanced) %>% 
    unnest(glanced)

regressions %>% 
    select(augmented) %>% 
    unnest(augmented)

# By combining the estimates and p-values across all groups into the same tidy
# data frame (instead of a list of output model objects), a new class of analyses
# and visualizations becomes straightforward. This includes:
#   sorting by p-value or estimate to find the most significant terms across all tests,
#   p-value histograms, and
#   volcano plots comparing p-values to effect size estimates.
# In each of these cases, we can easily filter, facet, or distinguish based on the
# term column. In short, this makes the tools of tidy data analysis available for
# the results of data analysis and models, not just the inputs.
