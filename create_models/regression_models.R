# create_models/regression_models.R
#
#
library(tidymodels)
library(janitor)

data(ames)
ames <- ames %>% 
    clean_names()

set.seed(4595)

data_split <- initial_split(ames, strata = sale_price,
                            prop = 0.75)

ames_train <- training(data_split)
ames_test <- testing(data_split)

#                       random forest
rf_defaults <- rand_forest(mode = 'regression')
rf_defaults

# The parsnip package provides two different interfaces to fit a model:
#   the formula interface (fit()), and
#   the non-formula interface (fit_xy()).

preds <- c('longitude', 'latitude', 'lot_area', 'neighborhood', 'year_sold')
rf_xy_fit <-
    rf_defaults %>% 
    set_engine('ranger') %>% 
    fit_xy(
        x = ames_train %>% select(preds),
        y = log10(ames_train %>%  select(sale_price))
    )
rf_xy_fit

# for regrssion models, we can use the basic `predict method, which returns
# a tibble with a column named `.pred`
test_results <-
    ames_test %>% 
    select(sale_price) %>% 
    mutate(sale_price = log10(sale_price)) %>% 
    bind_cols(predict(rf_xy_fit,
                      new_data = ames_test %>% select(preds)))
test_results

# summarize performance
test_results %>% 
    metrics(truth = sale_price, estimate = .pred)


# lets use the formula method using new param values
rand_forest(mode = 'regression', mtry = 3, trees = 1000) %>% 
    set_engine('ranger') %>% 
    fit(
        log10(sale_price) ~ longitude + latitude + lot_area + neighborhood + year_sold,
        data = ames_train
    )

# suppose that we would like to use tne randomForest package
# instead of ranger, to do so, the only part of the syntax that needs to change
# is the `set_engine` arg
rand_forest(mode = 'regression', mtry = 3, trees = 1000) %>% 
    set_engine('randomForest') %>% 
    fit(
        log10(sale_price) ~ longitude + latitude + lot_area + neighborhood + year_sold,
        data = ames_train
    )

# `.preds()` returns the number of predictor varialbes in teh data set
# that are associated wuth the predicotd prior to dummy variable creatioon
rand_forest(mode = 'regression', mtry = .preds(), trees = 1000) %>% 
    set_engine('ranger') %>% 
    fit(
        log10(sale_price) ~ longitude + latitude + lot_area + neighborhood + year_sold,
        data = ames_train
    )



#                               Regularized regression
# when regularization is used, the predictors should first be centered
# and scaled before being passed to the model. The forula method wont do that
# automatically so we will need to do this ourselves.
norm_recipe <-
    recipe(sale_price ~ longitude + latitude + lot_area + neighborhood + year_sold,
           data = ames_train) %>% 
    step_other(neighborhood) %>% 
    step_dummy(all_nominal()) %>% 
    step_center(all_predictors()) %>% 
    step_scale(all_predictors()) %>% 
    step_log(sale_price, base = 10) %>% 
    # estimate the means and std devs
    prep(training = ames_train, retain = TRUE)

# now lets fit the model using the processed version of the data
glmn_fit <-
    linear_reg(penalty = 0.001, mixture = 0.5) %>% 
    set_engine('glmnet') %>% 
    fit(sale_price ~ ., data = bake(norm_recipe, new_data = NULL))
glmn_fit

# If penalty were not specified, all of the lambda values would be computed.
# To get the predictions for this specific value of lambda (aka penalty):
# first, get the processed version of the test set predictors
test_normalized <-
    bake(norm_recipe, new_data = ames_test, all_predictors())

test_results <-
    test_results %>% 
    rename(`random_forest` = .pred) %>% 
    bind_cols(
        predict(glmn_fit, new_data = test_normalized) %>% 
            rename(glmnet = .pred)
    )
test_results

test_results %>% 
    metrics(truth = sale_price,
            estimate = glmnet)


test_results %>% 
    gather(model, prediction, -sale_price) %>% 
    ggplot(mapping = aes(x = prediction, y = sale_price)) +
    geom_abline(col = 'green', lty = 2) +
    geom_point(alpha = 0.4) +
    facet_wrap(~model) +
    coord_fixed()

# This final plot compares the performance of the random forest and regularized regression models.