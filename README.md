# Predictive Analysis: Predicting Credit Card Default


### TABLE OF CONTENTS
* [Abstract](#Abstract)
* [Introduction](#Introduction)
* [Related Work](#Related-Work)
* [Models](#Models)
* [Results](#Results)
* [Conclusion](#Conclusion)
* [References](#References)
* [Appendix A](#AppendixA)
* [Appendix B](#AppendixB)

## Abstract
In this project, 5 machine learning methods were performed to identify delinquency rates of credit card clients. Through the comparison of classification machine learning methods, the aim is to study the best method with the consideration of demographic features and financial aspects for each client. From the perspective of mitigating potential loss to financial institutions, model assessments were performed through performance metrics; in particular, focusing on recall, f1-score and accuracy of each model.

## Introduction
The delinquency rates of credit cards measured by the Federal Reserve Bank of New York have consistently experienced the highest credit card delinquencies over many years before the financial crisis. According to the Federal Reserve System, the credit card delinquencies over time are to be seen as encouraging and saw the largest decrease during the pandemic, going from 9.2% in the first quarter of 2020 down to 5.7% before 2021 due to the travel and entertainment restrictions [[1]](#References). While this seems to be a positive sign where credit card delinquencies remained well below the levels in comparison to early 2000, it would be seen as a temporary result of displaced discretionary expenses due to the COVID-19 restrictions. In addition, the possibility of using credit cards as an extension of their income would still be a social issue that lives beyond one’s means.

The purpose of this project is to determine the best model that can identify credit card clients that will most likely be defaulted based on a range of demographic and past payment’s history attributes. The goal is to use supervised machine learning algorithms to determine the most ideal machine learning model when implementing the algorithms. After this project, the following questions will be answered:

* What is the relationship between demographic variables and payment history?

* How to determine the default rate based on demographic information?

* What variables contribute the most to classifying default or non-default clients?

* Create a model to predict default with a level higher than the baseline prediction can provide?

As such, the optimal outcome of this project will be to apply the chosen final model to reduce the default rate via classifying the credit card clients accurately into default and non-default.


## Related-Work
According to Husejinovic et al.[[2]](#References), their way of conducting default payment prediction for eight machine learning models (logistic regression, C4.5, SVM, naïve bayes, k-NN, and ensemble learning methods) is to apply outliers and extreme values elimination based on interquartile range and perform feature selection through wrapper method named classifier subset evaluator. Studies from Bai proposed that outliers can be determined through density-based outlier detection techniques [[3]](#References). This is further experimented by Wang et al. where a two-layer ensemble model is being implemented for outlier detection [[4]](#References). In this project, the approach of feature engineering through standard scalar method for data normalization noise removal through interquartile range method might take away some potential useful values as it might just be referring to individuals that possessed more wealth.

Furthermore, data level resampling techniques are being explored to overcome the issues of data imbalance. From there, it can be beneficial to know if different resampling techniques would improve the proposed model performance. Kerdprasop and Kerdprasop [[5]](#References) performed random oversampling and SMOTE method to increase the accuracy of their learning model (regression model, SVM, decision tree, and neural network). Their findings show that SMOTE obtained the highest specificity model while random over sampling model has the highest sensitivity model. I perform the experiments mentioned in related works to arrive at the best estimation for the model to determine if my approach would outperform the findings from the related works and if not, what I should refer to and potentially improve the overall model accuracy from there.

## Models
**Logistic Regression**

The logistic regression model is a classification algorithm when the targeted data is categorical in nature. It is a statistical method for analyzing a dataset when the data has a binary or multinomial output, such as when it belongs to one class or another, or if it is either a 0 or 1 [[6]](#References). While it is easy to implement, it is limited when working with non-linear data. Often, logistic regression can have the tendency to overfit the training data, which becomes amplified when there is an increase in the training data. [[ps. 1]](#AppendixA)

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756051-89b18e73-7ecc-4c7e-bbf8-a8eafb6a3f8a.png" width="200" />

**Support Vector Machines (SVM)**

Support Vector Machine is a linear model that creates a hyperplane that directly separates the data into classes. This hyperplane is placed in a N-dimensional space, where N is the number of features. While logistic regression tends to focus more on maximizing the probability of two classes, SVM uses the hyperplane to maximize the separation of classes. Using the hyper-parameter, C, modifies the hyperplane’s margin to classify the training data properly. Increasing the value of C signifies more accurate training points [[7]](#References) [[ps. 2]](#AppendixA)

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756087-46745538-f7cf-4ffb-a136-3a729d2db192.png" width="265" />

**Decision Tree**

The decision tree algorithm works by building a classification model that results in a tree-like structure. It follows a principle of maximization of the separation of the data. This maximization starts with the training data on a single node, where it splits into two nodes. This split happens through learning simple decision rules deduced from the training data. This splitting continues until the decision tree achieves a leaf node that contains the predicted class value. To solve the issue of overfitting, the depth of the tree gets assigned a default value. [[7]](#References) [[ps. 3]](#AppendixA)

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756118-576eec5a-61a8-44ba-8c80-36968ab94e94.png" width="205" />

**Naive Bayes**

The Naive-Bayes algorithm is based on the Bayes theorem. This theorem assumes the independence of the predictor variables, which allows the algorithm to calculate the value of an attribute without affecting the others. [[8]](#References) It works by using characteristics and cases with large likelihood to calculate the probability of classification. [[ps. 4]](#AppendixA)

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756129-9f2d669c-4f60-4013-92f6-dc359fc665b1.png" width="225" />

**K-Nearest Neighbors (KNN)**

The K-Nearest Neighbors algorithm differs from the models in this list, by directly using the data for classification. It stores all the available data and classifies the new data based on the similarity amount, with data points classified based on how its neighbors are classified. The algorithm omits building a model first which provides the benefit of not requiring additional model construction; with k being the only adjustable parameter. [[9]](#References) The k represents the number of nearest neighbors to be included in the model. Therefore, through the value adjustment of k, the model can be made more or less flexible. [[ps. 5]](#AppendixA)

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756136-eeab8b14-cb0f-47c0-be4a-825024204236.png" width="425" />

**Performance Measures**

Before creating the models, splitting the dataset into a training set and testing set must be done. Further, a confusion matrix is built. This confusion matrix consists of the number of correct True Positives (TP), the number of correct predictions marked negative (TN), the number of incorrect predictions that are marked positive (FP), and the number of incorrect negative predictions (FN).

* The **Accuracy** determines the ratio of correct predictions over the overall predictions

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756145-2865e860-3dbd-4a4a-93bc-e010a07ea271.png" width="250" />

* The **Precision** determines how well the algorithm is able to find true positives.

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756156-303422af-6bf6-4400-b395-dda044fe0576.png" width="150" />

* The **Recall** determines how well the classification model can find all of the true positive samples.

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756165-21655773-6372-4172-a477-22aa1b6a043b.png" width="125" />

* The **F1** measure is the weighted mean between recall and precision. This F1 score helps with model determine how correct the positive predictions are

<p align="center"> <img src="https://user-images.githubusercontent.com/97916773/175756169-d9b0c623-15ef-4a92-977b-5dee0a8de85d.png" width="180" />



## Results
I used the multivariate classification dataset from UCI Machine Learning Repository. [[10]](#References) It contains 30,000 observations and 24 variables in total where it corresponds to an individual credit card client. The data variables can be divided into two categories: 23 original features as explanatory variables and one response variable where the default individual is indicated as 1 where the otherwise is 0.

**Dataset Initial Observations**

The original data types are all labeled as ‘Object’ and therefore data types for all numerical data variables are updated to the respective ‘int64’ and ‘float64’ for later EDA and machine learning purposes. There are no null values in this dataset, and it is important to find out if the attributes of all variables are valid. This would be addressed in the univariate analysis.

**Pre-processing Data**

* Renaming column ‘Pay_0’ to ‘Pay_1’ for consistency with the rest of the columns; ‘default payment next month’ to ‘default_payment_result’ for easier referencing [fig.1]

* Drop ‘ID’ column as it is not part of the modeling process [fig.1]

* Label categorical data ‘SEX’, ‘EDUCATION’, ‘MARRIAGE’ as categorical label for easier reference [fig. 2]

* Clean ‘EDUCATION’, ‘MARRIAGE’ i.e. group low counts categories in respective columns as ‘others’ to make sure it will not impact regression accuracy [fig. 3]

**EDA (Univariate & Multivariate Analysis)**

*Univariate Analysis*

* Considering ‘SEX’, ‘EUDCATION’, ‘MARRIAGE’, ‘PAY_1 through PAY_6’, ‘default_payment_result’ as categorical variables, a bar chart was created by categorizing them as a subset of the dataset to separate them with the continuous numeric variables E.g., noted that out of 30000 credit cards, 23364 were default credit cards which is 77.9% [fig. 4]
* Create sns.histplot and density plot for the rest of numeric variables [fig. 5]

*Multivariate Analysis*

* Performed multivariate analysis in relation to the target variable y = ‘default_payment_result’ via visualization of bar charts, scatter plots, and box plots to investigate the relationships against numerical and categorical features [fig. 6]

* Hypothesis tests are performed to determine if there is a relationship between default rate and categorical variables in specifically to Sex, Marital Status, Education Level H0: There is no relationship between categorical variables and default rate H1: There is a relationship between categorical variables and default rate

Noting that all three variables are statistically different from the default rate, we reject the null hypothesis for categorical variables and therefore they are included in the later model estimation.

**Split Data: Training and Testing Set**

The original cleaned dataset is divided into variable X and target variable Y into 70-30 training and testing sets for the baseline model and perform another set of training and testing data with standardized features. The purpose of standardization is to rescale the features where the mean and standard deviation would be 0 and 1 respectively. The result has been improved significantly after feature scaling as it helped to standardize the range of features where it affected the result heavily in computing the distance.

**Metric Selection**

Performed five metrics in this project - accuracy, precision, recall, F1-Score, and ROC to determine the performance of each machine learning model. It is important to focus more on the recall and accuracy as recall is the ability of the classifier to find positive class. This aligns to the project’s objective, where the cost associated with false negatives and hence recall is the important metric for the project to detect default clients (default rate =1).

**Baseline**

In this project, baseline models were performed to establish a foundation of the data. The main model was a simple mean baseline model, which took the mean of all the data and calculated the accuracy of predicting non-default. The mean baseline model was put under consideration however; the recall value is the main importance for the project to compare and improve on.

Therefore, in addition to the mean baseline model; Decision Tree, Naïve Bayes, KNearest Neighbors, and Logistic Regression baseline models were performed as a point of reference to compare to the tuned models.

**Feature Engineering**

To improve the baseline models, feature engineering is performed through the following methods:

* Multicollinearity Analysis: features with correlation coefficient >= 0.92 will be removed [fig. 7 and 8]

* Standard Scalar: perform standardization to transform their values with mean value of 0 and variance of 1 [fig. 9]

* One-Hot Encoding: Other than dropping ‘ID’, one-hot-encoding to categorical data is being conducted to multiple categorical columns and assigned them 0 or 1. The encoded variable allows the categorical data ‘SEX’, ‘EDUCATION’, and, ‘MARRIAGE’ [fig. 10]

**Model Estimation** 

Performed a series of model development approaches through the following steps:

1. Perform baseline models mentioned in the previous section (mean baseline model, Logistic Regression, Decision Tree, Naive Bayes, k-NN)

2. Perform GridSearch CV and hyperparameter tuning to improve recall rate and overall accuracy

3. Implement Stochastic Gradient Descent

4. Improve tuned models through oversampling and undersampling training data

5. Review model result through performance metrics in specifically to ROC and PR curve

6. Propose the best performance model for credit card default rate classification


| Baseline Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | ---| --- |
| Decision Tree | 0.72 | 0.38 | 0.40 | 0.39 | 0.61 |
| Naive Bayes | 0.40 | 0.25 | 0.85 | 0.38 | 0.66 |
| K-NN | 0.75 | 0.37 | 0.18 | 0.24 | 0.61 |
| Logistic Regression | 0.78 | 0.25 | 0 | 0 | 0.64 |
<p align="center"><sup>Table 1: Baseline Model Results </sup></p>

Based on the table above, the ROC curve and bar chart are plotted to visualize which model has the highest accuracy [fig. 11] . It is important to note that the Logistic Regression baseline method has the highest accuracy but with 0 recall. Knowing that Recall rate is the main metric to focus on, Naive Bayes has the highest performance among all of the baseline models as it obtains 0.85 recall rate. Next is hyperparameter tuning for models to experiment different hyper-parameters and to find the best combination of hyperparameter tuning for each model.

**Model Estimation after Hyperparameter Tuning**
  
The approach of performing hyperparameter tuning would be using the GridSearchCV package where the best parameters are selected to tune the respective models after setting a range of parameters. Then plotting ROC [fig. 12] and PR curve [fig. 13] based on the performance result.
  
| Grid Search | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | ---| --- |
| SVM (C = 1, kernel = rbf) | 0.82 | 0.66 | 0.35 | 0.46 | 0.65 |
| K-NN (leaf size = 1, k = 14, weights = uniform) | 0.81 | 0.65 | 0.31 | 0.42 | 0.63 |
| Decision Tree (max depth = 4, min sample leaf = 6, min sample split = 2) | 0.82 | 0.66 | 0.36 | 0.46 | 0.65 |
| Logistic Regression (C = 0.1, penalty = l2) | 0.81 | 0.67 | 0.32 | 0.43 | 0.64 |
| Naive Bayes | 0.80 | 0.57 | 0.39 | 0.46 | 0.65 |
| Stochastic Gradient Descent | 0.82 | 0.70 | 0.31 | 0.43 | 0.64 |
<p align="center"><sup>Table 2: Model Results after Hyperparameter Tuning </sup></p>

While there are some improvements in logistic regression and k-NN, logistic regression’s recall is still not performing as expected with a recall rate of 0.32. I wanted to know if oversampling and undersampling of training data would help to improve recall rate.
  
**Model Estimation after Undersampling and Oversampling**
  
Random oversampling and random undersampling are performed based on hyperparameter tuning results. 77.9% of non default clients and 22.1% default clients are randomly sampled:
  
|  | Before Random Undersampling | After Random Undersampling | Before Random Oversampling | After Random Oversampling |
| --- | --- | --- | --- | ---|
| **Default Rate = 1** | 4645 | 4645 | 4645 | 16355 |
| **Default Rate = 0** | 16355 | 4645 | 16355 | 16355 | 
<p align="center"><sup>Table 3: Undersampling / Oversampling dataset </sup></p>
  
**Oversampling**

The overall recall rate are being improved after performing sampling method:

| Oversampling | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | ---| --- |
| SVM | 0.76 | 0.46 | 0.55 | 0.50 | 0.68 |
| K-NN | 0.73 | 0.39 | 0.39 | 0.39 | 0.61 |
| Decision Tree | 0.74 | 0.44 | 0.57 | 0.49 | 0.68 |
| Logistic Regression | 0.78 | 0.50 | 0.55 | 0.52 | 0.69 |
| Naive Bayes | 0.75 | 0.44 | 0.57 | 0.50 | 0.68 |
| Stochastic Gradient Descent | 0.77 | 0.47 | 0.57 | 0.52 | 0.69 |
<p align="center"><sup>Table 4: Model Results after Random Oversampling Method </sup></p>

Through a random oversampling method, Stochastic Gradient Descent performs the best in terms of recall value 0.57, followed by Decision Tree and Naive Bayes.

**Undersampling**
| Undersampling | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | ---| --- |
| SVM | 0.75 | 0.45 | 0.60 | 0.51 | 0.70 |
| K-NN | 0.75 | 0.44 | 0.57 | 0.50 | 0.68 |
| Decision Tree | 0.77 | 0.48 | 0.55 | 0.51 | 0.69 |
| Logistic Regression | 0.78 | 0.49 | 0.55 | 0.52 | 0.69 |
| Naive Bayes | 0.75 | 0.44 | 0.57 | 0.50 | 0.68 |
| Stochastic Gradient Descent | 0.78 | 0.50 | 0.54 | 0.52 | 0.69 |
<p align="center"><sup>Table 5: Model Results after Random Undersampling Method </sup></p>

Similar results were shown after performing a random undersampling method for all models. However, SVM and K-NN showed the best improvement in recall value with this method. Here SVM performs the best in terms of recall value 0.60, followed by Naive Bayes and K-NN.

## Conclusion

A structural model development approach was performed to model the delinquency rates of credit card problems to determine clients that would most likely go into default. Through a simple method of dummy classification, hyperparameter tuning improved the overall model performance, especially in accuracy rate. However, the overall recall rate for all models after hyperparameter tuning are not performing well with the average of 0.34. To determine if the overall recall rate can be improved further, undersampling and oversampling was performed on training data and the recall rate has been improved to an average of 0.55. In addition, I found that the best default rate predictors from this dataset are "amount paid in previous months", "amount of given credit", and "repayment status in previous months".

As part of a future research, I would like to continue performing different methods to increase recall of our models through PCA, further algorithm tuning or perform different methods of oversampling and undersampling e.g. SMOTE, Cluster Centroid. Also, a potential idea is to look into setting specific objective functions to redirect the project to fit the macroeconomic circumstances instead of relying fully on this dataset. For example, financial institutions might have more risk appetite and may be more inclined to minimize false negatives and less geared towards minimizing false positives. Hence, by setting a specific cost function, it would be more practical to reflect the current macroeconomic situation.


## References
(1) Board of governors of the Federal Reserve System. The Fed - Consumer Credit - G.19. (n.d.). [cited October 29, 2021], from https://www.federalreserve.gov/releases/g19/current/.

[2] A. Husejinovic, D. Kečo, and Z. Mašetić. Application of Machine Learning Algorithms in Credit Card Default Payment Prediction. International Journal of Scientific Research. 7. 425.10.15373/22778179#husejinovic, 2018.

[3] M. Bai, X. Wang, J. Xin, and G. Wang, An efficient algorithm for distributed density-based outlier detection on big data, Neurocomputing, pg. 19-28 ISSN 0925-2312, doi:10.1016/j.neucom.2015.05.135., 2016.

[4] S. Wei, D. Yang, W. Zhang and S. Zhang, A Novel Noise-Adapted Two-Layer Ensemble Model for Credit Scoring Based on Backflow Learning, in IEEE Access, vol. 7, pp. 99217-99230, doi:10.1109/ACCESS.2019.2930332., 2019.

[5] N. Kerdprasop and K. Kerdprasop, On the Generation of Accurate Predictive Model from Highly Imbalanced Data with Heuristics and replication Technologies, International Journal of Bio-Science and Bio-Technology, vol. 4, pg. 49- 64, 2012.

[6] A. Worster, J. Fan, and A. Ismaila. Understanding Linear and Logistic Regression Analyses. CJEM. pg. 112–3. 2007.

[7] K.S. Naik. Predicting Credit Risk for Unsecured Lending: A Machine Learning Approach. NMIMS MPSTME, pg. 5. 2021.

[8] Ajay, A. Venkatesh, and S.G. Jacob. Prediction of Credit-Card Defaulters: A Comparative Study on Performance of Classifiers. International Journal of Computer Applications (0975-8887), vol. 145, pg 38. 2016.

[9] B.V. Dasarathy. Nearest Neighbor Pattern Classification Techniques. Computer Society Press, 1990.

[10] I. Yeh and C. Lien, The Comparisons of Data Mining Techniques for the Predictive Accuracy of Probability of Default of Credit Card Clients, Expert Systems with Applications, vol. 36, no. 2, pp. 2473-2480, 2009. Appen


## AppendixA

## AppendixB
