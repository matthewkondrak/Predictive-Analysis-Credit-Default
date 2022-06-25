# Predictive Analysis: Predicting Credit Card Default


### TABLE OF CONTENTS
* [Abstract](#Abstract)
* [Introduction](#Introduction)
* [Related Work](#Related-Work)
* [Model](#Model)
* [Results](#Results)
* [Conclusion](#Conclusion)

## Abstract
In this study, 5 machine learning methods were performed to identify delinquency rates of credit card clients. Through the comparison of classification machine learning methods, the aim is to study the best method with the consideration of demographic features and financial aspects for each client. From the perspective of mitigating potential loss to financial institutions, model assessments were performed through performance metrics; in particular, focusing on recall, f1-score and accuracy of each model.

## Introduction
The delinquency rates of credit cards measured by the Federal Reserve Bank of New York have consistently experienced the highest credit card delinquencies over many years before the financial crisis. According to the Federal Reserve System, the credit card delinquencies over time are to be seen as encouraging and saw the largest decrease during the pandemic, going from 9.2% in the first quarter of 2020 down to 5.7% before 2021 due to the travel and entertainment restrictions [1]. While this seems to be a positive sign where credit card delinquencies remained well below the levels in comparison to early 2000, it would be seen as a temporary result of displaced discretionary expenses due to the COVID-19 restrictions. In addition, the possibility of using credit cards as an extension of their income would still be a social issue that lives beyond one’s means.

The purpose of this project is to determine the best model that can identify credit card clients that will most likely be defaulted based on a range of demographic and past payment’s history attributes. The goal is to use supervised machine learning algorithms to determine the most ideal machine learning model when implementing the algorithms. After this project, the following questions will be answered:

* What is the relationship between demographic variables and payment history?

* How to determine the default rate based on demographic information?

* What variables contribute the most to classifying default or non-default clients?

* Create a model to predict default with a level higher than the baseline prediction can provide?

As such, the optimal outcome of this project will be to apply the chosen final model to reduce the default rate via classifying the credit card clients accurately into default and non-default.


## Related-Work
According to Husejinovic et al.[2], their way of conducting default payment prediction for eight machine learning models (logistic regression, C4.5, SVM, naïve bayes, k-NN, and ensemble learning methods) is to apply outliers and extreme values elimination based on interquartile range and perform feature selection through wrapper method named classifier subset evaluator. Studies from Bai proposed that outliers can be determined through density-based outlier detection techniques [3]. This is further experimented by Wang et al. where a two-layer ensemble model is being implemented for outlier detection [4]. In this study, the approach of feature engineering through standard scalar method for data normalization noise removal through interquartile range method might take away some potential useful values as it might just be referring to individuals that possessed more wealth.

Furthermore, data level resampling techniques are being explored to overcome the issues of data imbalance. From there, it can be beneficial to know if different resampling techniques would improve the proposed model performance. Kerdprasop and Kerdprasop [5] performed random oversampling and SMOTE method to increase the accuracy of their learning model (regression model, SVM, decision tree, and neural network). Their findings show that SMOTE obtained the highest specificity model while random over sampling model has the highest sensitivity model. I perform the experiments mentioned in related works to arrive at the best estimation for the model to determine if my approach would outperform the findings from the related works and if not, what I should refer to and potentially improve the overall model accuracy from there.

## Model
**Logistic Regression**

The logistic regression model is a classification algorithm when the targeted data is categorical in nature. It is a statistical method for analyzing a dataset when the data has a binary or multinomial output, such as when it belongs to one class or another, or if it is either a 0 or 1 [6]. While it is easy to implement, it is limited when working with non-linear data. Often, logistic regression can have the tendency to overfit the training data, which becomes amplified when there is an increase in the training data. [ps. 1]

**Support Vector Machines**

Support Vector Machine is a linear model that creates a hyperplane that directly separates the data into classes. This hyperplane is placed in a N-dimensional space, where N is the number of features. While logistic regression tends to focus more on maximizing the probability of two classes, SVM uses the hyperplane to maximize the separation of classes. Using the hyper-parameter, C, modifies the hyperplane’s margin to classify the training data properly. Increasing the value of C signifies more accurate training points. [7] [ps. 2]

**Decision Tree**

The decision tree algorithm works by building a classification model that results in a tree-like structure. It follows a principle of maximization of the separation of the data. This maximization starts with the training data on a single node, where it splits into two nodes. This split happens through learning simple decision rules deduced from the training data. This splitting continues until the decision tree achieves a leaf node that contains the predicted class value. To solve the issue of overfitting, the depth of the tree gets assigned a default value. [7] [ps. 3]

**Naive Bayes**

The Naive-Bayes algorithm is based on the Bayes theorem. This theorem assumes the independence of the predictor variables, which allows the algorithm to calculate the value of an attribute without affecting the others. [8] It works by using characteristics and cases with large likelihood to calculate the probability of classification. [ps. 4]

**K-Nearest Neighbors (KNN)**

The K-Nearest Neighbors algorithm differs from the models in this list, by directly using the data for classification. It stores all the available data and classifies the new data based on the similarity amount, with data points classified based on how its neighbors are classified. The algorithm omits building a model first which provides the benefit of not requiring additional model construction; with k being the only adjustable parameter. [9] The k represents the number of nearest neighbors to be included in the model. Therefore, through the value adjustment of k, the model can be made more or less flexible. [ps. 5]

**Performance Measures**

Before creating the models, splitting the dataset into a training set and testing set must be done. Further, a confusion matrix is built. This confusion matrix consists of the number of correct True Positives (TP), the number of correct predictions marked negative (TN), the number of incorrect predictions that are marked positive (FP), and the number of incorrect negative predictions (FN).

The **Accuracy** determines the ratio of correct predictions over the overall predictions

The **Precision** determines how well the algorithm is able to find true positives.

The **Recall** determines how well the classification model can find all of the true positive samples.

The **F1** measure is the weighted mean between recall and precision. This F1 score helps with model determine how correct the positive predictions are

## Results


## Conclusion

## References
[1] Board of governors of the Federal Reserve System. The Fed - Consumer Credit - G.19. (n.d.). [cited October 29, 2021], from https://www.federalreserve.gov/releases/g19/current/.
