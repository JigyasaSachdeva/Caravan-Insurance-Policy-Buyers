# Caravan-Insurance-Policy-Buyers
The data is taken from Kaggle’s 2000 challenge. The dataset is owned and supplied by Dutch data mining company Sentiment Machine Research and is a real world business problem. The problem statement is to classify customers as potential buyers of the Caravan Insurance Policy based on historical data of existing buyers. The dataset comprises customer’s socio-demographic, product ownership and insurance statistics data.
The following are the files and the descriptions: 
- IDS575_JigyasaSachdeva_FinalProject.R: This R file contains data loading, pre-processing, exploratory data analysis, hypothesis testing, writing SMOTE and encoded data, creating baseline model, developing logistic regression with cut off tuning, Support vector machine with cost tuning and Random Forest with mtree tuning
- KNN.ipynb: K Nearest Neighbors was implemented on the train data to make predictions on test data. Hyperparameter 'k' was tuned using hold out cross validation
- RandomForest.ipynb: Tuned mtree was fed to grid search cross validation to tune other parameters: Number of features, Number of trees, Maximum depth, Minimum samples required for splitting, Minimum samples required in leaf, Bootstrap
- LogisticRegression.ipynb: Final model development with tuning parameters Class weights, Penalty (Ridge/Lasso), Intercept Fit, C (Inverse of regularization Strength) to evaluate ridge regression's fit 
- ModelComparison.ipynb: Comparing all the models with baseline using Precision Recall Curves and F score metric table. 
- Final_Presentation.pptx: Presentation encapsulating the project's summary
- Final_Report.pdf: Report paper enlisting details of the approaches took and the rationale behind them
