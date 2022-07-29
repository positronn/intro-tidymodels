# getting_started/tune_model_parameters.R
#
#

# Some model parameters cannot be learned directly from a data set during model training;
# these kinds of parameters are called hyperparameters. Some examples of hyperparameters
# include the number of predictors that are sampled at splits in a tree-based model (we call
# this mtry in tidymodels) or the learning rate in a boosted tree model
# (we call this learn_rate). Instead of learning these kinds of hyperparameters during model
# training, we can estimate the best values for these values by training many models on
# resampled data sets and exploring how well all these models perform. This process
# is called tuning.

library(tidymodels)
# Helper packages
library(rpart.plot)  # for visualizing a decision tree
library(vip)         # for variable importance plots

data(cells, package = "modeldata")
cells

#                               * Predicting image segmentation, but better *
#
# Random forest models are a tree-based ensemble method, and typically perform well with
# default hyperparameters. However, the accuracy of some other tree-based models, such as
# boosted tree models or decision tree models, can be sensitive to the values of hyperparameters.
# In this article, we will train a decision tree model. There are several hyperparameters
# for decision tree models that can be tuned for better performance. 
# Tuning these hyperparameters can improve model performance because decision tree models
# are prone to overfitting. This happens because single tree models tend to fit the training
# data too well — so well, in fact, that they over-learn patterns present in the training data
# that end up being detrimental when predicting new data.

# We will tune the model hyperparameters to avoid overfitting. Tuning the value of cost_complexity
# helps by pruning back our tree. It adds a cost, or penalty, to error rates of
# more complex trees; a cost closer to zero decreases the number tree nodes pruned and is
# more likely to result in an overfit tree. However, a high cost increases the number of tree
# nodes pruned and can result in the opposite problem—an underfit tree. Tuning tree_depth, on
# the other hand, helps by stopping our tree from growing after it reaches a certain depth.
# We want to tune these hyperparameters to find what those two values should be for our model to
# do the best job predicting image segmentation.

# Before we start the tuning process, we split our data into training and testing sets,
# just like when we trained the model with one default set of hyperparameters. As before,
# we can use strata = class if we want our training and testing sets to be created using
# stratified sampling so that both have the same proportion of both kinds of segmentation.
set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

# We use the training data for tuning the model.

#                           * Tuning hyperparameters *
# Let’s start with the parsnip package, using a decision_tree() model with the rpart engine.
# To tune the decision tree hyperparameters cost_complexity and tree_depth, we create a model
# specification that identifies which hyperparameters we plan to tune.
tune_spec <- 
    decision_tree(
        cost_complexity = tune(),
        tree_depth = tune()
    ) %>% 
    set_engine("rpart") %>% 
    set_mode("classification")

tune_spec

# Think of tune() here as a placeholder. After the tuning process, we will select a single
# numeric value for each of these hyperparameters. For now, we specify our parsnip model
# object and identify the hyperparameters we will tune().
# We can’t train this specification on a single data set (such as the entire training set)
# and learn what the hyperparameter values should be, but we can train many models using
# resampled data and see which models turn out best. We can create a regular grid of
# values to try using some convenience functions for each hyperparameter:
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)
# The function grid_regular() is from the dials package. It chooses sensible values to
# try for each hyperparameter; here, we asked for 5 of each. Since we have two to tune,
# grid_regular() returns 5 × 5 = 25 different possible tuning combinations to try in a
# tidy tibble format.
tree_grid

# Here, you can see all 5 values of cost_complexity ranging up to 0.1. These values get
# repeated for each of the 5 values of tree_depth:
tree_grid %>% 
    count(tree_depth)
# Armed with our grid filled with 25 candidate decision tree models, let’s
# create cross-validation folds for tuning:
set.seed(234)
cell_folds <- vfold_cv(cell_train)
# Tuning in tidymodels requires a resampled object created with the rsample package.

#                           * Model tuning with a grid *
#
# ere we use a workflow() with a straightforward formula; if this model required
# more involved data preprocessing, we could use add_recipe() instead of add_formula().
set.seed(345)

tree_wf <- workflow() %>%
    add_model(tune_spec) %>%
    add_formula(class ~ .)

tree_res <- 
    tree_wf %>% 
    tune_grid(
        resamples = cell_folds,
        grid = tree_grid
    )

tree_res

# Once we have our tuning results, we can both explore them through visualization and
# then select the best result. The function collect_metrics() gives us a tidy tibble
# with all the results. We had 25 candidate models and two metrics, accuracy and roc_auc,
# and we get a row for each .metric and model.
tree_res %>% 
    collect_metrics()
# We might get more out of plotting these results:
tree_res %>%
    collect_metrics() %>%
    mutate(tree_depth = factor(tree_depth)) %>%
    ggplot(aes(cost_complexity, mean, color = tree_depth)) +
    geom_line(size = 1.5, alpha = 0.6) +
    geom_point(size = 2) +
    facet_wrap(~ .metric, scales = "free", nrow = 2) +
    scale_x_log10(labels = scales::label_number()) +
    scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

# We can see that our “stubbiest” tree, with a depth of 1, is the worst model
# according to both metrics and across all candidate values of cost_complexity.
# Our deepest tree, with a depth of 15, did better. However, the best tree seems to
# be between these values with a tree depth of 4. The show_best() function shows us
# the top 5 candidate models by default:
tree_res %>%
    show_best("accuracy")

# We can also use the select_best() function to pull out the single set of
# hyperparameter values for our best decision tree model:
best_tree <- tree_res %>%
    select_best("accuracy")

best_tree

# We can update (or “finalize”) our workflow object tree_wf with the values from select_best().
final_wf <- 
    tree_wf %>% 
    finalize_workflow(best_tree)

final_wf

#                               * The last fit *
#
# Finally, let’s fit this final model to the training data and use our test
# data to estimate the model performance we expect to see with new data. We
# can use the function last_fit() with our finalized model; this function fits the
# finalized model on the full training data set and evaluates the finalized model on
# the testing data.
final_fit <- 
    final_wf %>%
    last_fit(cell_split) 

final_fit %>%
    collect_metrics()

final_fit %>%
    collect_predictions() %>% 
    roc_curve(class, .pred_PS) %>% 
    autoplot()
# The performance metrics from the test set indicate that we did not overfit during our tuning procedure.

final_tree <- extract_workflow(final_fit)
final_tree

# The final_fit object contains a finalized, fitted workflow that you can use for
# predicting on new data or further understanding the results. You may want to
# extract this object, using one of the extract_ helper functions.

# We can create a visualization of the decision tree using another helper function
# to extract the underlying engine-specific fit.
final_tree %>%
    extract_fit_engine() %>%
    rpart.plot(roundint = FALSE)

final_tree %>% 
    extract_fit_parsnip() %>% 
    vip()
