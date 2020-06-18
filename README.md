# heavy-tail-regression

Author: Emmanuel Ekwedike

Date: 1/19/2019

Gaussian linear models are often insufficient in practical applications, where noise can be heavy-tailed. We consider a simple linear model, where the conditional mean of the error term given the independent variable is zero. The goal is to derive a good estimator for the parameters of the linear model based on the observed sample data. 

We build two models: Weighted Least Square and Huber Estimate to derive a good estimator for the parameters of the linear model using four different data samples. 

/data/: the folder containing datasets used for model building.

main.py: the python script containing code.

regression_models.py: the python module containing code for the regression models.

main.ipynb: the jupyter notebook containing code.

results.csv: the spreadsheet containing the estimated parameters of the linear model.