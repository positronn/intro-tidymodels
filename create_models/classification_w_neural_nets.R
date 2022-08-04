# create_models/classification_w_neural_nets.R
#
# Classification models using a neural network
# 
library(tidymodels)
library(keras)

data(bivariate)
dim(bivariate_train)
dim(bivariate_test)
dim(bivariate_val)

# a plot of the data shows two right-skewed predictos
ggplot(bivariate_train, aes(x = A, y = B, col = Class)) +
    geom_point(alpha = 0.2)

# Let’s use a single hidden layer neural network to predict
# the outcome. To do this, we transform the predictor columns
# to be more symmetric (via the step_BoxCox() function) and on a
# common scale (using step_normalize()). We can use recipes to do so:
biv_rec <-
    recipe(Class ~ ., data = bivariate_train) %>% 
    step_BoxCox(all_predictors()) %>% 
    step_normalize(all_predictors()) %>% 
    prep(training = bivariate_train, retain = TRUE)

# we will bake(new_data = NULL) to get the processed trining set back
# for validation
val_normalized <-
    bake(biv_rec, new_data = bivariate_val, all_predictors())
# for testing when we arrive at a final model
test_normalized <-
    bake(biv_rec, new_data = bivariate_test, all_predictors())

# We can use the keras package to fit a model with 5 hidden units and a
# 10% dropout rate, to regularize the model:
set.seed(57974)
nnet_fit <-
    mlp(epochs = 100, hidden_units = 2, dropout = 0.1) %>%
    set_mode("classification") %>% 
    # Also set engine-specific `verbose` argument to prevent logging the results: 
    set_engine("keras", verbose = 0) %>%
    fit(Class ~ ., data = bake(biv_rec, new_data = NULL))

nnet_fit


#                               Model performance
#
# In parsnip, the predict() function can be used to characterize performance on the validation set.
# Since parsnip always produces tibble outputs, these can just be column bound to the original data:
val_results <- 
    bivariate_val %>%
    bind_cols(
        predict(nnet_fit, new_data = val_normalized),
        predict(nnet_fit, new_data = val_normalized, type = "prob")
    )
val_results %>% slice(1:5)

val_results %>%
    roc_auc(truth = Class, .pred_One)

val_results %>%
    accuracy(truth = Class, .pred_class)

val_results %>%
    conf_mat(truth = Class, .pred_class)



a_rng <- range(bivariate_train$A)
b_rng <- range(bivariate_train$B)
x_grid <-
    expand.grid(A = seq(a_rng[1], a_rng[2], length.out = 100),
                B = seq(b_rng[1], b_rng[2], length.out = 100))
x_grid_trans <- bake(biv_rec, x_grid)

# Make predictions using the transformed predictors but 
# attach them to the predictors in the original units: 
x_grid <- 
    x_grid %>% 
    bind_cols(predict(nnet_fit, x_grid_trans, type = "prob"))

ggplot(x_grid, aes(x = A, y = B)) + 
    geom_contour(aes(z = .pred_One), breaks = .5, col = "black") + 
    geom_point(data = bivariate_val, aes(col = Class), alpha = 0.3)