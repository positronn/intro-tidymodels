# getting_started/evaluate_model_w_resampling.R
#
# So far, we have built a model and preprocessed data with a recipe. We also
# introduced workflows as a way to bundle a parsnip model and recipe together.
# Once we have a model trained, we need a way to measure how well that model
# predicts new data. This tutorial explains how to characterize model performance
# based on resampling statistics.
# To use code in this article, you will need to install the following
# packages: modeldata, ranger, and tidymodels.

library(tidymodels)

# Helper packages
library(modeldata)  # for the cells data

# Let’s use data from Hill, LaPan, Li, and Haney (2007), available in the
# modeldata package, to predict cell image segmentation quality with
# resampling. To start, we load this data into R:
data(cells, package = "modeldata")
cells

# We have data for 2019 cells, with 58 variables. The main outcome variable
# of interest for us here is called class, which you can see is a factor.
# But before we jump into predicting the class variable, we need to understand
# it better. Below is a brief primer on cell image segmentation.

# The cells data has class labels for 2019 cells — each cell is labeled as
# either poorly segmented (PS) or well-segmented (WS). Each also has a total of 56
# predictors based on automated image analysis measurements. For example, avg_inten_ch_1
# is the mean intensity of the data contained in the nucleus, area_ch_1 is the total
# size of the cell, and so on (some predictors are fairly arcane in nature).

# The rates of the classes are somewhat imbalanced; there are more poorly
# segmented cells than well-segmented cells:
cells %>% 
    count(class) %>% 
    mutate(prop = n/sum(n))

# In our previous Preprocess your data with recipes article, we started by
# splitting our data. It is common when beginning a modeling project to separate
# the data set into two partitions:
# The training set is used to estimate parameters, compare models and feature
# engineering techniques, tune models, etc.
# The test set is held in reserve until the end of the project, at which point
# there should only be one or two models under serious consideration. It
# is used as an unbiased source for measuring final model performance.

# There are different ways to create these partitions of the data. The most
# common approach is to use a random sample. Suppose that one quarter of the data
# were reserved for the test set. Random sampling would randomly select 25% for the
# test set and use the remainder for the training set. We can use the rsample
# package for this purpose.

# Since random sampling uses random numbers, it is important to set the random
# number seed. This ensures that the random numbers can be reproduced at a later
# time (if needed).

# The function rsample::initial_split() takes the original data and saves the
# information on how to make the partitions. In the original analysis,
# the authors made their own training/test set and that information is
# contained in the column case. To demonstrate how to make a split, we’ll
# remove this column before we make our own split:
set.seed(123)
cell_split <-
    initial_split(cells %>% select(-case), 
                            strata = class)

#Here we used the strata argument, which conducts a stratified split. This
# ensures that, despite the imbalance we noticed in our class variable, our training
# and test data sets will keep roughly the same proportions of poorly and
# well-segmented cells as in the original data. After the initial_split, the
# training() and testing() functions return the actual data sets.
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

nrow(cell_train)
nrow(cell_train)/nrow(cells)

# training set proportions by class
cell_train %>% 
    count(class) %>% 
    mutate(prop = n/sum(n))

cell_test %>% 
    count(class) %>% 
    mutate(prop = n/sum(n))

# The majority of the modeling work is then conducted on the training set data.


# Random forest models are ensembles of decision trees. A large number of decision tree
# models are created for the ensemble based on slightly different versions of the training set.
# When creating the individual decision trees, the fitting process encourages them to be as diverse
# as possible. The collection of trees are combined into the random forest model and, when
# a new sample is predicted, the votes from each tree are used to calculate the final predicted
# value for the new sample. For categorical outcome variables like class in our cells data example,
# the majority vote across all the trees in the random forest determines the predicted class
# for the new sample.

# To fit a random forest model on the training set, let’s use the parsnip package
# with the ranger engine. We first define the model that we want to create:
rf_mod <- 
    rand_forest(trees = 1000) %>% 
    set_engine("ranger") %>% 
    set_mode("classification")

# Starting with this parsnip model object, the fit() function can be used with a model
# formula. Since random forest models use random numbers, we again set
# the seed prior to computing:
set.seed(234)
rf_fit <- 
    rf_mod %>% 
    fit(class ~ ., data = cell_train)
rf_fit

# This new rf_fit object is our fitted model, trained on our training data set.


#                           * Estimating performance *
#
# During a modeling project, we might create a variety of different models.
# To choose between them, we need to consider how well these models do, as measured by
# some performance statistics. In our example in this article, some options we could use are:
#       the area under the Receiver Operating Characteristic (ROC) curve, and 
#       overall classification accuracy. 

# The ROC curve uses the class probability estimates to give us a sense of performance
# across the entire set of potential probability cutoffs. Overall accuracy uses the hard
# class predictions to measure performance. The hard class predictions tell us whether
# our model predicted PS or WS for each cell. But, behind those predictions, the model is
# actually estimating a probability. A simple 50% probability cutoff is used to
# categorize a cell as poorly segmented.
# The yardstick package has functions for computing both of these measures called roc_auc() and accuracy().

# At first glance, it might seem like a good idea to use the training set data
# to compute these statistics. (This is actually a very bad idea.) Let’s see what happens
# if we try this. To evaluate performance based on the training set, we call the predict()
# method to get both types of predictions (i.e. probabilities and hard class predictions).
rf_training_pred <- 
    predict(rf_fit, cell_train) %>% 
    bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
    # Add the true outcome data back in
    bind_cols(cell_train %>% 
                  select(class))

rf_training_pred %>%                # training set predictions
    roc_auc(truth = class, .pred_PS)

rf_training_pred %>%                # training set predictions
    accuracy(truth = class, .pred_class)


# Now that we have this model with exceptional performance, we proceed to the test set.
# Unfortunately, we discover that, although our results aren’t bad, they are certainly worse
# than what we initially thought based on predicting the training set:
rf_testing_pred <- 
    predict(rf_fit, cell_test) %>% 
    bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
    bind_cols(cell_test %>% select(class))

rf_testing_pred %>%                   # test set predictions
    roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%                   # test set predictions
    accuracy(truth = class, .pred_class)

# 
# There are several reasons why training set statistics like the ones shown in this
# section can be unrealistically optimistic:
#     
#     Models like random forests, neural networks, and other black-box methods can
#       essentially memorize the training set. Re-predicting that same set should always
#       result in nearly perfect results.
# 
#     The training set does not have the capacity to be a good arbiter of performance.
#       It is not an independent piece of information; predicting the training set can
#       only reflect what the model already knows.
# 
#     To understand that second point better, think about an analogy from teaching.
#       Suppose you give a class a test, then give them the answers, then provide
#       the same test. The student scores on the second test do not accurately reflect
#       what they know about the subject; these scores would probably be higher than
#       their results on the first test.


#                           * Resampling *
# 
# Resampling methods, such as cross-validation and the bootstrap, are empirical simulation
# systems. They create a series of data sets similar to the training/testing split
# discussed previously; a subset of the data are used for creating the model and a different
# subset is used to measure performance. Resampling is always used with the training set. 

# Let’s use 10-fold cross-validation (CV) in this example. This method randomly allocates
# the 1514 cells in the training set to 10 groups of roughly equal size, called “folds”.
# For the first iteration of resampling, the first fold of about 151 cells are held out for
# the purpose of measuring performance. This is similar to a test set but, to avoid confusion,
# we call these data the assessment set in the tidymodels framework.

# The other 90% of the data (about 1362 cells) are used to fit the model. Again, this
# sounds similar to a training set, so in tidymodels we call this data the analysis set.
# This model, trained on the analysis set, is applied to the assessment set to generate
# predictions, and performance statistics are computed based on those predictions.

# In this example, 10-fold CV moves iteratively through the folds and leaves a different
# 10% out each time for model assessment. At the end of this process, there are 10 sets
# of performance statistics that were created on 10 data sets that were not used in the
# modeling process. For the cell example, this means 10 accuracies and 10 areas under the
# ROC curve. While 10 models were created, these are not used further; we do not keep the
# models themselves trained on these folds because their only purpose is calculating
# performance metrics.


# To generate these results, the first step is to create a resampling object using rsample.
# There are several resampling methods implemented in rsample; cross-validation folds can
# be created using vfold_cv():
set.seed(345)
folds <- vfold_cv(cell_train, v = 10)
folds


# The list column for splits contains the information on which rows belong in the
# analysis and assessment sets. There are functions that can be used to extract the
# individual resampled data called analysis() and assessment().

# However, the tune package contains high-level functions that can do the required
# computations to resample a model for the purpose of measuring performance. You have
# several options for building an object for resampling:
#    Resample a model specification preprocessed with a formula or recipe, or 
#    Resample a workflow() that bundles together a model specification and formula/recipe. 

# For this example, let’s use a workflow() that bundles together the random forest model
# and a formula, since we are not using a recipe. Whichever of these options you use, the
# syntax to fit_resamples() is very similar to fit():
rf_wf <- 
    workflow() %>%
    add_model(rf_mod) %>%
    add_formula(class ~ .)

set.seed(456)
rf_fit_rs <- 
    rf_wf %>% 
    fit_resamples(folds)

# The results are similar to the folds results with some extra columns.
# The column .metrics contains the performance statistics created from the 10 assessment
# sets. These can be manually unnested but the tune package contains a number of simple
# functions that can extract these data:
collect_metrics(rf_fit_rs)

# Think about these values we now have for accuracy and AUC. These performance
# metrics are now more realistic (i.e. lower) than our ill-advised first attempt at
# computing performance metrics in the section above. If we wanted to try different
# model types for this data set, we could more confidently compare performance metrics
# computed using resampling to choose between models. Also, remember that at the end of our project,
# we return to our test set to estimate final model performance. We have looked at this
# once already before we started using resampling, but let’s remind ourselves of the results:
rf_testing_pred %>%                   # test set predictions
    roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%                   # test set predictions
    accuracy(truth = class, .pred_class)


# The performance metrics from the test set are much closer to the performance
# metrics computed using resampling than our first (“bad idea”) attempt. Resampling
# allows us to simulate how well our model will perform on new data, and the test set
# acts as the final, unbiased check for our model’s performance.

