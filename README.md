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

Standard Scalint (Yes, No)

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
Model building and prediction for a set of introduced values.

The available algorithms are the same that were mentioned in "Models".
It also supports multiclass problems.

Other available options:
- Standard Scaling
- Use the predictors with an introduced value or use all the predictors.


*Multiclass Classification not available


