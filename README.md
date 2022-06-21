# Predictive Analysis: Predicting Credit Card Default

###### Disclaimer

### TABLE OF CONTENTS
* [Abstract](#Abstract)
* [Introduction](#Introduction)
* [Related Work](#Related Work)
* [Model](#Model)
* [Experimental Results](#Experimental Results)
* [Conclusion](#Conclusion)

## Abstract
In this study, 5 machine learning methods were performed to identify delinquency rates of credit card clients. Through comparison of classification machine learning methods, the aim is to study the best method with the consideration of demographic features and financial aspects for each client. From the perspective of mitigating potential loss to financial institutions, model assessments were performed through performance metrics; in particular, focusing on recall, f1-score and accuracy of each model. Naive Bayes performs the best after performing feature engineering and hyperparameter tuning through the grid search method. To increase the overall metric performance of each model, oversampling and undersampling methods shall be performed as it drastically improves the model performance and from this Naive Bayes has the highest recall rate after oversampling.

## Introduction
The delinquency rates of credit cards measured by the Federal Reserve Bank of New York have consistently experienced the highest credit card delinquencies over many years before the financial crisis. According to the Federal Reserve System, the credit card delinquencies over time are to be seen as encouraging and saw the largest decrease during the pandemic, going from 9.2% in the first quarter of 2020 down to 5.7% before 2021 due to the travel and entertainment restrictions [1]. While this seems to be a positive sign where credit card delinquencies remained well below the levels in comparison to early 2000, it would be seen as a temporary result of displaced discretionary expenses due to the COVID-19 restrictions. In addition, the possibility of using credit cards as an extension of their income would still be a social issue that lives beyond one’s means.

The purpose of this project is to determine the best model that can identify credit card clients that will most likely be defaulted based on a range of demographic and past payment’s history attributes. The direction of this project will be using supervised machine learning algorithms to determine the most ideal machine learning model when implementing the algorithms. We will be answering the following questions:

* What is the relationship between demographic variables and payment history?

* How do we determine the default rate based on demographic information?

* What variables contribute the most to classifying default or non-default clients?

* Can we create a model to predict default with a level higher than the baseline prediction can provide?

As such, the optimal outcome of this project will be to apply the chosen final model to reduce the default rate via classifying the credit card clients accurately into default and non-default.

## Related Work

