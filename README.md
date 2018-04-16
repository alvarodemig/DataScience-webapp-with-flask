# DataScience webapp with Flask
Data Science webapp to show some of the capabilities of Flask and libraries such as sklearn, pandas, matplotlib, seaborn...

## Capabilities:

### Dataset upload
The webapp supports:
- CSV (delimiter: ',')
- TXT (delitimer: tab)

### Dataset summary
This page will show:
- 5 first rows to see the general aspect of the dataset.
- Statistical summary of each column.

### Preprocessing
We can create a new dataset (it will be saved as CSV) with the following options:

Feature selection:
- Automatic selection based on Chi-Squared estimator (the name will be created depending of the chosen parameters):
  - Number of features
  - Response variable
- Manual selection:
  - Name of the new dataset
  - Variables selection

Null values and columns with a unique value:
- Drop rows with null values:
  - Null in ALL columns
  - Null in ANY column
  - Never
- Drop variables with a unique value:
  - Yes
  - No
  
* Extra preprocessing (normalization, dummy variables...) will be done in model and predict steps.

### Graphs
Available visualizations for the chosen variables:

- Histograms
- BoxPlots
- Correlation plots

### Models
Models for Classification and Regression tasks.
It does not support multiclass classification at this moment (extra code to manage some metrics and graphs)

Available Algorithms:
- Logistic Regression (Classification)
- Linear Regression (Regression)
- Random Forests (both)
- K Nearest Neighbors (both)
- AdaBoost (both)
- Extreme Gradient Boosting (both)
- MultiLayer Perceptron (both)

K-Fold Cross-Validation (3, 5, 10)

Standard Scaling (Yes, No)

Manual Feature Selection

Classification Tasks Output:
- Fit time
- Score time
- Precision (Test and Train)
- Recall (Test and Train)
- F1 score (Test and Train)
- Accuracy (Test and Train)
- ROC AUC (Test and Train)
- ROC curves plot

Regression Tasks Output:
- Fit time
- Score time
- Explained Variance (Test and Train)
- R2 (Test and Train)
- Mean Squared Error (Test and Train)
- Measured vs Predicted values plot

### Predictions
Model building (with the complete dataset) and prediction for a set of introduced values.
The model will only include the variables with an introduced value.
The available algorithms are the same that were mentioned in "Models".
It also supports multiclass problems.

-------------------------------------------------------------------

Some ideas for improvement:
- Add formats and delimiters
- More feature estimators
- Possibility to choose between Train/test splits and Cross-Validation
- Add Clustering Algorithms.
- Parameter tuning
- Multiclass classification in "Models"
- Save model results in a database
- Predict using all the columns, filling the empty variables with the mean or other estimator (this would only work for numeric variables).
- Output personalization
- Customized error to give more information about happened (trying to predict a categorical variable with a regression  algorithm, etc.)
- Upload a file with data to predict.
- Dataset shape and number of categorical and numeric variables.
...

### VIDEO DEMONSTRATION
[![LINK TO YOUTUBE VIDEO](https://github.com/alvarodemig/DataScience-webapp-with-flask/blob/master/YoutubeLinkPicture.JPG)](https://www.youtube.com/watch?v=BxizdTrItTk)
