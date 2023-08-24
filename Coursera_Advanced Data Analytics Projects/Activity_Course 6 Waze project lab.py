#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 6 - The nuts and bolts of machine learning**

# Your team is close to completing their user churn project. Previously, you completed a project proposal, and used Python to explore and analyze Waze’s user data, create data visualizations, and conduct a hypothesis test. Most recently, you built a binomial logistic regression model based on multiple variables.
# 
# Leadership appreciates all your hard work. Now, they want your team to build a machine learning model to predict user churn. To get the best results, your team decides to build and test two tree-based models: random forest and XGBoost.
# 
# Your work will help leadership make informed business decisions to prevent user churn, improve user retention, and grow Waze’s business.
# 

# # **Course 6 End-of-Course Project: Build a machine learning model**
# 
# In this activity, you will practice using tree-based modeling techniques to predict on a binary target class.
# <br/>
# 
# **The purpose** of this model is to find factors that drive user churn.
# 
# **The goal** of this model is to predict whether or not a Waze user is retained or churned.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Build a machine learning model**
# 

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   What are you being asked to do?
# 
# 
# 2.   What are the ethical implications of the model? What are the consequences of your model making errors?
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a Waze user won't churn, but they actually will)?
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a Waze user will churn, but they actually won't)?
# 
# 3.  Do the benefits of such a model outweigh the potential problems?
# 4.  Would you proceed with the request to build this model? Why or why not?
# 
# 

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-4 HERE

# ### **Task 1. Imports and data loading**
# 
# Import packages and libraries needed to build and evaluate random forest and XGBoost classification models.

# In[ ]:


# Import packages for data manipulation
import pandas as pd 
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# This lets us see all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# Import packages for data modeling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# This is the function that helps plot feature importance
from sklearn.inspection import permutation_importance

# This module lets us save our models once we fit them.
import pickle


# Now read in the dataset as `df0` and inspect the first five rows.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[ ]:


# Import dataset
df0 = pd.read_csv('waze_dataset.csv')


# In[ ]:


# Inspect the first five rows
print(df0.head())


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2. Feature engineering**
# 
# You have already prepared much of this data and performed exploratory data analysis (EDA) in previous courses. You know that some features had stronger correlations with churn than others, and you also created some features that may be useful.
# 
# In this part of the project, you'll engineer these features and some new features to use for modeling.
# 
# To begin, create a copy of `df0` to preserve the original dataframe. Call the copy `df`.

# In[ ]:


# Copy the df0 dataframe
### YOUR CODE HERE ###


# Call `info()` on the new dataframe so the existing columns can be easily referenced.

# In[ ]:


df0.info()


# #### **`km_per_driving_day`**
# 
# 1. Create a feature representing the mean number of kilometers driven on each driving day in the last month for each user. Add this feature as a column to `df`.
# 
# 2. Get descriptive statistics for this new feature
# 
# 

# In[ ]:


# 1. Create `km_per_driving_day` feature
df['km_per_driving_day'] = df['km_driven_last_month'] / df['driving_days'] 

# 2. Get descriptive stats
print(df['km_per_driving_day'].describe())


# Notice that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[ ]:


# 1. Convert infinite values to zero
df['km_per_driving_day'] = df['km_per_driving_day'].replace(np.inf, 0)

# 2. Confirm that it worked
print(df['km_per_driving_day'].describe())


# #### **`percent_sessions_in_last_month`**
# 
# 1. Create a new column `percent_sessions_in_last_month` that represents the percentage of each user's total sessions that were logged in their last month of use.
# 
# 2. Get descriptive statistics for this new feature

# In[ ]:


# 1. Create `percent_sessions_in_last_month` feature
df['percent_sessions_in_last_month'] = df['sessions_in_last_month'] / df['total_sessions']

# 1. Get descriptive stats
print(df['percent_sessions_in_last_month'].describe())


# #### **`professional_driver`**
# 
# Create a new, binary feature called `professional_driver` that is a 1 for users who had 60 or more drives <u>**and**</u> drove on 15+ days in the last month.
# 
# **Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

# To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
# 1. A condition
# 2. What to return when the condition is true
# 3. What to return when the condition is false
# 
# ```
# Example:
# x = [1, 2, 3]
# x = np.where(x > 2, 100, 0)
# x
# array([  0,   0, 100])
# ```

# In[ ]:


# Create `professional_driver` feature
df['professional_driver'] = np.where((df['driving_days'] >= 60) & (df['drives_per_day'] >= 15), 1, 0)


# #### **`total_sessions_per_day`**
# 
# Now, create a new column that represents the mean number of sessions per day _since onboarding_.

# In[ ]:


# Create `total_sessions_per_day` feature
df['total_sessions_per_day'] = df['total_sessions'] / df['days_since_onboard']


# As with other features, get descriptive statistics for this new feature.

# In[ ]:


# Get descriptive stats
print(df['total_sessions_per_day'].describe())


# #### **`km_per_hour`**
# 
# Create a column representing the mean kilometers per hour driven in the last month.

# In[ ]:


# Create `km_per_hour` feature
df['km_per_hour'] = df['km_driven_last_month'] / df['hours_driven_last_month']


# #### **`km_per_drive`**
# 
# Create a column representing the mean number of kilometers per drive made in the last month for each user. Then, print descriptive statistics for the feature.

# In[ ]:


# Create `km_per_drive` feature
df['km_per_drive'] = df['km_driven_last_month'] / df['drives_per_day']




# This feature has infinite values too. Convert the infinite values to zero, then confirm that it worked.

# In[ ]:


# 1. Convert infinite values to zero
df['km_per_drive'] = df['km_per_drive'].replace(np.inf, 0)


# 2. Confirm that it worked
print(df['km_per_drive'].describe())



# #### **`percent_of_sessions_to_favorite`**
# 
# Finally, create a new column that represents the percentage of total sessions that were used to navigate to one of the users' favorite places. Then, print descriptive statistics for the new column.
# 
# This is a proxy representation for the percent of overall drives that are to a favorite place. Since total drives since onboarding are not contained in this dataset, total sessions must serve as a reasonable approximation.
# 
# People whose drives to non-favorite places make up a higher percentage of their total drives might be less likely to churn, since they're making more drives to less familiar places.

# In[ ]:


# Create `percent_of_sessions_to_favorite` feature
df['percent_of_sessions_to_favorite'] = df['favorite_place_sessions'] / df['total_sessions']


# Get descriptive stats
print(df['percent_of_sessions_to_favorite'].describe())



# ### **Task 3. Drop missing values**
# 
# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[ ]:


# Drop rows with missing values
df.dropna(inplace=True)


# ### **Task 4. Outliers**
# 
# You know from previous EDA that many of these columns have outliers. However, tree-based models are resilient to outliers, so there is no need to make any imputations.

# ### **Task 5. Variable encoding**

# #### **Dummying features**
# 
# In order to use `device` as an X variable, you will need to convert it to binary, since this variable is categorical.
# 
# In cases where the data contains many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Each possible category of each feature will result in a feature for your model, which could lead to an inadequate ratio of features to observations and/or difficulty understanding your model's predictions.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[ ]:


# Create new `device2` variable
df['device2'] = np.where(df['device'] == 'Android', 0, 1)



# #### **Target encoding**
# 
# The target variable is also categorical, since a user is labeled as either "churned" or "retained." Change the data type of the `label` column to be binary. This change is needed to train the models.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` so as not to overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[ ]:


# Create binary `label2` column
df['label2'] = np.where(df['label'] == 'retained', 0, 1)



# ### **Task 6. Feature selection**
# 
# Tree-based models can handle multicollinearity, so the only feature that can be cut is `ID`, since it doesn't contain any information relevant to churn.
# 
# Note, however, that `device` won't be used simply because it's a copy of `device2`.
# 
# Drop `ID` from the `df` dataframe.

# In[ ]:


# Drop `ID` column
df.drop(['ID'], axis=1, inplace=True)



# ### **Task 7. Evaluation metric**
# 
# Before modeling, you must decide on an evaluation metric. This will depend on the class balance of the target variable and the use case of the model.
# 
# First, examine the class balance of your target variable.

# In[ ]:


# Get class balance of 'label' col
print(df['label'].value_counts(normalize=True))



# Approximately 18% of the users in this dataset churned. This is an unbalanced dataset, but not extremely so. It can be modeled without any class rebalancing.
# 
# Now, consider which evaluation metric is best. Remember, accuracy might not be the best gauge of performance because a model can have high accuracy on an imbalanced dataset and still fail to predict the minority class.
# 
# It was already determined that the risks involved in making a false positive prediction are minimal. No one stands to get hurt, lose money, or suffer any other significant consequence if they are predicted to churn. Therefore, select the model based on the recall score.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 8. Modeling workflow and model selection process**
# 
# The final modeling dataset contains 14,299 samples. This is towards the lower end of what might be considered sufficient to conduct a robust model selection process, but still doable.
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 
# Note that, when deciding the split ratio and whether or not to use a validation set to select a champion model, consider both how many samples will be in each data partition, and how many examples of the minority class each would therefore contain. In this case, a 60/20/20 split would result in \~2,860 samples in the validation set and the same number in the test set, of which \~18%&mdash;or 515 samples&mdash;would represent users who churn.
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

# ### **Task 9. Split the data**
# 
# Now you're ready to model. The only remaining step is to split the data into features/target variable and training/validation/test sets.
# 
# 1. Define a variable `X` that isolates the features. Remember not to use `device`.
# 
# 2. Define a variable `y` that isolates the target variable (`label2`).
# 
# 3. Split the data 80/20 into an interim training set and a test set. Don't forget to stratify the splits, and set the random state to 42.
# 
# 4. Split the interim training set 75/25 into a training set and a validation set, yielding a final ratio of 60/20/20 for training/validation/test sets. Again, don't forget to stratify the splits and set the random state.

# In[ ]:


# 1. Isolate X variables
X = df.drop(['label', 'label2'], axis=1)

# 2. Isolate y variable
y = df['label2']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 4. Split into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) 



# Verify the number of samples in the partitioned data.

# In[ ]:

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# This aligns with expectations.

# ### **Task 10. Modeling**

# #### **Random forest**
# 
# Begin with using `GridSearchCV` to tune a random forest model.
# 
# 1. Instantiate the random forest classifier `rf` and set the random state.
# 
# 2. Create a dictionary `cv_params` of any of the following hyperparameters and their corresponding values to tune. The more you tune, the better your model will fit the data, but the longer it will take.
#  - `max_depth`
#  - `max_features`
#  - `max_samples`
#  - `min_samples_leaf`
#  - `min_samples_split`
#  - `n_estimators`
# 
# 3. Define a dictionary `scoring` of scoring metrics for GridSearch to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `rf_cv`. Pass to it as arguments:
#  - estimator=`rf`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit=_`)
# 
#  `refit` should be set to `'recall'`.<font/>
# 
# 

# **Note:** To save time, this exemplar doesn't use multiple values for each parameter in the grid search, but you should include a range of values in your search to home in on the best set of parameters.

# In[ ]:


# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5], 
             'max_features': [6],
             'min_samples_leaf': [100], 
             'n_estimators': [200]}


# 3. Define a dictionary of scoring metrics to capture
scoring = {'precision': make_scorer(precision_score), 
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'accuracy': make_scorer(accuracy_score)}

# 4. Instantiate the GridSearchCV object
rf_cv = GridSearchCV(estimator=rf, 
                    param_grid=cv_params,
                    scoring=scoring,
                    cv=5,
                    refit='recall')


# Now fit the model to the training data.

# In[ ]:

rf_cv.fit(X_train, y_train)


# Examine the best average score across all the validation folds.

# In[ ]:


# Examine best score
print(rf_cv.best_score_)


# Examine the best combination of hyperparameters.

# In[ ]:


# Examine best hyperparameter combo
print(rf_cv.best_params_)


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To learn more about how this function accesses the cross-validation results, refer to the [`GridSearchCV` scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) for the `cv_results_` attribute.
# 
# </details>

# In[ ]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

  # Create dictionary that maps input metric to actual metric name in GridSearchCV
 metric_map = {'precision': 'mean_test_precision',
                  'recall': 'mean_test_recall',
                  'f1': 'mean_test_f1',
                  'accuracy': 'mean_test_accuracy'}

  # Get all the results from the CV and put them in a df
   results = model_object.cv_results_
    best_params = model_object.best_params_
    best_index = model_object.best_index_
     df_results = pd.DataFrame(results)
        
  # Isolate the row of the df with the max(metric) score
     best_row = df_results.iloc[best_index]

  # Extract Accuracy, precision, recall, and f1 score from that row
  
    accuracy = best_row[metric_map['accuracy']]
    precision = best_row[metric_map['precision']] 
    recall = best_row[metric_map['recall']]
    f1 = best_row[metric_map['f1']]

  # Create table of results
table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                         })
    
    return table


# Pass the `GridSearch` object to the `make_results()` function.

# In[ ]:


print(make_results('RF', rf_cv, 'recall'))


# Asside from the accuracy, the scores aren't that good. However, recall that when you built the logistic regression model in the last course the recall was \~0.09, which means that this model has 33% better recall and about the same accuracy, and it was trained on less data.
# 
# If you want, feel free to try retuning your hyperparameters to try to get a better score. You might be able to marginally improve the model.

# #### **XGBoost**
# 
#  Try to improve your scores using an XGBoost model.
# 
# 1. Instantiate the XGBoost classifier `xgb` and set `objective='binary:logistic'`. Also set the random state.
# 
# 2. Create a dictionary `cv_params` of the following hyperparameters and their corresponding values to tune:
#  - `max_depth`
#  - `min_child_weight`
#  - `learning_rate`
#  - `n_estimators`
# 
# 3. Define a dictionary `scoring` of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `xgb_cv`. Pass to it as arguments:
#  - estimator=`xgb`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit='recall'`)

# In[ ]:


# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)


# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5],
             'learning_rate': [0.1],  
             'n_estimators': [200],
             'min_child_weight': [1]}

# 3. Define a dictionary of scoring metrics to capture
scoring = {'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score), 
           'f1': make_scorer(f1_score),
           'accuracy': make_scorer(accuracy_score)}

# 4. Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(estimator=xgb,
                      param_grid=cv_params, 
                      scoring=scoring,
                      cv=5,
                      refit='recall') 


# Now fit the model to the `X_train` and `y_train` data.
# 
# Note this cell might take several minutes to run.

# In[ ]:

xgb_cv.fit(X_train, y_train)


# Get the best score from this model.

# In[ ]:


# Examine best score
print(xgb_cv.best_score_)


# And the best parameters.

# In[ ]:


# Examine best parameters
print(xgb_cv.best_params_)


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.

# In[ ]:


# Call 'make_results()' on the GridSearch object
print(make_results('XGB', xgb_cv, 'recall'))


# This model fit the data even better than the random forest model. The recall score is nearly double the recall score from the logistic regression model from the previous course, and it's almost 50% better than the random forest model's recall score, while maintaining a similar accuracy and precision score.

# ### **Task 11. Model selection**
# 
# Now, use the best random forest model and the best XGBoost model to predict on the validation data. Whichever performs better will be selected as the champion model.

# #### **Random forest**

# In[ ]:


# Use random forest model to predict on validation data
rf_best = rf_cv.best_estimator_
rf_val_preds = rf_best.predict(X_val)


# Use the `get_test_scores()` function to generate a table of scores from the predictions on the validation data.

# In[ ]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table


# In[ ]:


# Get validation scores for RF model
rf_val_scores = get_test_scores('RF', rf_val_preds, y_val)

# Append to the results table
results = results.append(rf_val_scores)


# Notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.

# #### **XGBoost**
# 
# Now, do the same thing to get the performance scores of the XGBoost model on the validation data.

# In[ ]:


# Use XGBoost model to predict on validation data
xgb_best = xgb_cv.best_estimator_  
xgb_val_preds = xgb_best.predict(X_val)

# Get validation scores for XGBoost model
xgb_val_scores = get_test_scores('XGB', xgb_val_preds, y_val)

# Append to the results table
results = results.append(xgb_val_scores)


# Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 12. Use champion model to predict on test data**
# 
# Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.

# In[ ]:


# Use XGBoost model to predict on test data
xgb_test_preds = xgb_best.predict(X_test)

# Get test scores for XGBoost model
xgb_test_scores = get_test_scores('XGB Test', xgb_test_preds, y_test) 


# Append to the results table
results = results.append(xgb_test_scores)


# The recall was exactly the same as it was on the validation data, but the precision declined notably, which caused all of the other scores to drop slightly. Nonetheless, this is stil within the acceptable range for performance discrepancy between validation and test scores.

# ### **Task 13. Confusion matrix**
# 
# Plot a confusion matrix of the champion model's predictions on the test data.

# In[ ]:


# Generate array of values for confusion matrix
cm = confusion_matrix(y_test, xgb_test_preds)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")


# The model predicted three times as many false negatives than it did false positives, and it correctly identified only 16.6% of the users who actually churned.

# ### **Task 14. Feature importance**
# 
# Use the `plot_importance` function to inspect the most important features of your final model.

# In[ ]:


plot_importance(xgb, max_num_features=10)
plt.savefig('feature_importance.png')


# The XGBoost model made more use of many of the features than did the logistic regression model from the previous course, which weighted a single feature (`activity_days`) very heavily in its final prediction.
# 
# If anything, this underscores the importance of feature engineering. Notice that engineered features accounted for six of the top 10 features (and three of the top five). Feature engineering is often one of the best and easiest ways to boost model performance.
# 
# Also, note that the important features in one model might not be the same as the important features in another model. That's why you shouldn't discount features as unimportant without thoroughly examining them and understanding their relationship with the dependent variable, if possible. These discrepancies between features selected by models are typically caused by complex feature interactions.
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 
# Even if you cannot use the model to make strong predictions, was the work done in vain? What insights can you report back to stakeholders?

# ### **Task 15. Conclusion**
# 
# Now that you've built and tested your machine learning models, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. Would you recommend using this model for churn prediction? Why or why not?
# 
# 2. What tradeoff was made by splitting the data into training, validation, and test sets as opposed to just training and test sets?
# 
# 3. What is the benefit of using a logistic regression model over an ensemble of tree-based models (like random forest or XGBoost) for classification tasks?
# 
# 4. What is the benefit of using an ensemble of tree-based models like random forest or XGBoost over a logistic regression model for classification tasks?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?

# 1. # I would recommend using this model for churn prediction because it achieved a decent recall score of 0.334 on the test set, meaning it was able to correctly identify 33% of users who churned. The model has room for improvement, but it provides a good starting point for identifying potential churners.

# 2. By using a validation set, we lose some training examples that could have improved the model performance on the test set. However, it allows us to properly assess model performance during development and select the best model, avoiding overfitting to the test set.

# 3. Logistic regression is faster to train and simpler to interpret. The coefficients give insight into the relationship between features and the target. Tree ensembles can achieve better predictive performance but are more complex.

# 4. Tree ensembles like random forest and XGBoost tend to achieve better predictive performance, especially with noisy or complex data, by averaging across many tree models to reduce variance. They are also robust to outliers.

# 5. # Ideas to improve the model:
# - Try different combinations of hyperparameters 
# - Engineer additional features
# - Use class weights to handle imbalance
# - Try reducing overfitting with regularization 
# - Look for better quality training data

# 6.  More user demographic data, app usage/interaction data, customer support interactions, etc. could help find stronger signals related to churn.

# ### **BONUS**
# 
# The following content is not required, but demonstrates further steps that you might take to tailor your model to your use case.

# #### **Identify an optimal decision threshold**
# 
# The default decision threshold for most implementations of classification algorithms&mdash;including scikit-learn's&mdash;is 0.5. This means that, in the case of the Waze models, if they predicted that a given user had a 50% probability or greater of churning, then that user was assigned a predicted value of `1`&mdash;the user was predicted to churn.
# 
# With imbalanced datasets where the response class is a minority, this threshold might not be ideal. You learned that a precision-recall curve can help to visualize the trade-off between your model's precision and recall.
# 
# Here's the precision-recall curve for the XGBoost champion model on the test data.

# In[ ]:


# Plot precision-recall curve



# As recall increases, precision decreases. But what if you determined that false positives aren't much of a problem? For example, in the case of this Waze project, a false positive could just mean that a user who will not actually churn gets an email and a banner notification on their phone. It's very low risk.
# 
# So, what if instead of using the default 0.5 decision threshold of the model, you used a lower threshold?
# 
# Here's an example where the threshold is set to 0.4:

# In[ ]:


# Get predicted probabilities on the test data
y_probs = xgb_best.predict_proba(X_test)


# The `predict_proba()` method returns a 2-D array of probabilities where each row represents a user. The first number in the row is the probability of belonging to the negative class, the second number in the row is the probability of belonging to the positive class. (Notice that the two numbers in each row are complimentary to each other and sum to one.)
# 
# You can generate new predictions based on this array of probabilities by changing the decision threshold for what is considered a positive response. For example, the following code converts the predicted probabilities to {0, 1} predictions with a threshold of 0.4. In other words, any users who have a value ≥ 0.4 in the second column will get assigned a prediction of `1`, indicating that they churned.

# In[ ]:


# Create a list of just the second column values (probability of target)
probs = [row[1] for row in y_probs]

# Create an array of new predictions that assigns a 1 to any value >= 0.4
thresholds = np.arange(0, 1, 0.001)
scores = []

# In[ ]:


# Get evaluation metrics for when the threshold is 0.4
for threshold in thresholds:

     preds = [1 if prob >= threshold else 0 for prob in probs]

          recall = recall_score(y_test, preds)

              scores.append((threshold, recall))
distances = []


# Compare these numbers with the results from earlier.

# In[ ]:


for idx, score in enumerate(scores):
distance = abs(score[1] - desired_recall)

      distances.append((idx, distance))

sorted_distances = sorted(distances, key=lambda x: x[1])
  
best = sorted_distances[0]

best_idx = best[0]

threshold, recall = scores[best_idx]



# Recall and F1 score increased significantly, while precision and accuracy decreased.
# 
# So, using the precision-recall curve as a guide, suppose you knew that you'd be satisfied if the model had a recall score of 0.5 and you were willing to accept the \~30% precision score that comes with it. In other words, you'd be happy if the model successfully identified half of the people who will actually churn, even if it means that when the model says someone will churn, it's only correct about 30% of the time.
# 
# What threshold will yield this result? There are a number of ways to determine this. Here's one way that uses a function to accomplish this.

# In[ ]:


def threshold_finder(y_test_data, probabilities, desired_recall):
    '''
    Find the threshold that most closely yields a desired recall score.

    Inputs:
        y_test_data: Array of true y values
        probabilities: The results of the `predict_proba()` model method
        desired_recall: The recall that you want the model to have

    Outputs:
        threshold: The threshold that most closely yields the desired recall
        recall: The exact recall score associated with `threshold`
    '''
    probs = [x[1] for x in probabilities]  # Isolate second column of `probabilities`
    thresholds = np.arange(0, 1, 0.001)    # Set a grid of 1,000 thresholds to test

    scores = []
    for threshold in thresholds:
        # Create a new array of {0, 1} predictions based on new threshold
        preds = np.array([1 if x >= threshold else 0 for x in probs])
        # Calculate recall score for that threshold
        recall = recall_score(y_test_data, preds)
        # Append the threshold and its corresponding recall score as a tuple to `scores`
        scores.append((threshold, recall))

    distances = []
    for idx, score in enumerate(scores):
        # Calculate how close each actual score is to the desired score
        distance = abs(score[1] - desired_recall)
        # Append the (index#, distance) tuple to `distances`
        distances.append((idx, distance))

    # Sort `distances` by the second value in each of its tuples (least to greatest)
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=False)
    # Identify the tuple with the actual recall closest to desired recall
    best = sorted_distances[0]
    # Isolate the index of the threshold with the closest recall score
    best_idx = best[0]
    # Retrieve the threshold and actual recall score closest to desired recall
    threshold, recall = scores[best_idx]

    return threshold, recall


# Now, test the function to find the threshold that results in a recall score closest to 0.5.

# In[ ]:


# Get the predicted probabilities from the champion model
threshold, recall = threshold_finder(y_test, y_probs, 0.5)
print(threshold, recall)

# Call the function
probs = [row[1] for row in y_probs]

preds = [1 if prob >= threshold else 0 for prob in probs]


# Setting a threshold of 0.124 will result in a recall of 0.503.
# 
# To verify, you can repeat the steps performed earlier to get the other evaluation metrics for when the model has a threshold of 0.124. Based on the precision-recall curve, a 0.5 recall score should have a precision of \~0.3.

# In[ ]:


# Create an array of new predictions that assigns a 1 to any value >= 0.124
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
accuracy = accuracy_score(y_test, preds)

# Get evaluation metrics for when the threshold is 0.124
print(precision, recall, f1, accuracy)


# It worked! Hopefully now you understand that changing the decision threshold is another tool that can help you achieve useful results from your model.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
