# create_models/subsampling_for_class_imbalance.R
#
# Subsampling for class imbalances

# Subsampling a training set, either undersampling or oversampling the appropriate class or
# classes, can be a helpful approach to dealing with classification data where one or more
# classes occur very infrequently. In such a situation (without compensating for it),
# most models will overfit to the majority class and produce very good statistics for the class
# containing the frequently occurring classes while the minority classes have poor performance.
library(tidymodels)
library(readr)
library(discrim)
library(klaR)
library(ROSE)
library(themis)

# Consider a two-class problem where the first class has a very low rate of occurrence.
# The data were simulated and can be imported into R using the code below:
imbal_data <- 
    readr::read_csv("https://bit.ly/imbal_data") %>% 
    mutate(Class = factor(Class))
dim(imbal_data)
table(imbal_data$Class)

# If “Class1” is the event of interest, it is very likely that a classification model
# would be able to achieve very good specificity since almost all of the data are of
# the second class. Sensitivity, however, would likely be poor since the models will
# optimize accuracy (or other loss functions) by predicting everything to be the majority class.

# One result of class imbalance when there are two classes is that the default
# probability cutoff of 50% is inappropriate; a different cutoff that is more extreme
# might be able to achieve good performance.


imbal_rec <- 
    recipe(Class ~ ., data = imbal_data) %>%
    step_rose(Class)

# For a model, let’s use a quadratic discriminant analysis (QDA) model. From the
# discrim package, this model can be specified using:
qda_mod <- 
    discrim_regularized(frac_common_cov = 0, frac_identity = 0) %>% 
    set_engine("klaR")

qda_rose_wflw <- 
    workflow() %>% 
    add_model(qda_mod) %>% 
    add_recipe(imbal_rec)
qda_rose_wflw


# Model performance
set.seed(5732)
cv_folds <- vfold_cv(imbal_data, strata = "Class", repeats = 5)

# If a model is poorly calibrated, the ROC curve value might not show diminished performance.
# However, the J index would be lower for models with pathological distributions for the
# class probabilities. The yardstick package will be used to compute these metrics.
cls_metrics <- metric_set(roc_auc, j_index)

# Now, we train the models and generate the results using tune::fit_resamples():
set.seed(2180)
qda_rose_res <- fit_resamples(
    qda_rose_wflw, 
    resamples = cv_folds, 
    metrics = cls_metrics
)

collect_metrics(qda_rose_res)

# What do the results look like without using ROSE? We can create another workflow and
# fit the QDA model along the same resamples:
qda_wflw <- 
    workflow() %>% 
    add_model(qda_mod) %>% 
    add_formula(Class ~ .)

set.seed(2180)
qda_only_res <- fit_resamples(qda_wflw, resamples = cv_folds, metrics = cls_metrics)
collect_metrics(qda_only_res)

# It looks like ROSE helped a lot, especially with the J-index. Class imbalance
# sampling methods tend to greatly improve metrics based on the hard class predictions
# (i.e., the categorical predictions) because the default cutoff tends to be a better
# balance of sensitivity and specificity.

# Let’s plot the metrics for each resample to see how the individual results changed.
no_sampling <- 
    qda_only_res %>% 
    collect_metrics(summarize = FALSE) %>% 
    dplyr::select(-.estimator) %>% 
    mutate(sampling = "no_sampling")

with_sampling <- 
    qda_rose_res %>% 
    collect_metrics(summarize = FALSE) %>% 
    dplyr::select(-.estimator) %>% 
    mutate(sampling = "rose")

bind_rows(no_sampling, with_sampling) %>% 
    mutate(label = paste(id2, id)) %>%  
    ggplot(aes(x = sampling, y = .estimate, group = label)) + 
    geom_line(alpha = .4) + 
    facet_wrap(~ .metric, scales = "free_y")

# This visually demonstrates that the subsampling mostly affects metrics that
# use the hard class predictions.