# Credit_Risk_Analysis

## Overview of the analysis:

This project uses machine learning to evaluate credit card risk and help classify high-risk and low-risk loans. Credit cards typically have an unbalanced classification where low-risk loans outnumber high-risk loans. Hence, the project uses different techniques to train and evaluate models with unbalanced classes. The project uses `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling. 

Using the dataset from LendingClub, a lending service company, the data will first be oversampled using `RandomOverSample` and `SMOTE` algorithms. Second, the data will be undersampled using the `ClusterCentroids` algorithm. Then, the combination approach of over and under sampling is utilized along with the `SMOTEENN` algorithm. Next, two machine learning models that reduce bias are compared, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk.

## Results: 

#### Oversampling - Naive Random Oversampling
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/Oversampling.png "Title")
- The Random Oversampling accuracy score was .6367. Meaning the `y_pred` model was accurate 63.67% of the time. 
- The precision score measures how reliable a model predicts a positive classification. The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .01. 
- The recall score for high-risk loans was 62% and 65% for low-risk loans. Recall tests the ability of a model to find all the positive samples. These scores indicate a relatively large number of false negatives, and the model does a poor job of identifying high-risk loans. 

#### Oversampling - SMOTE
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/SMOTE.png "Title")
- The Random Oversampling accuracy score was .6302. Meaning the `y_pred` model was accurate 63.02% of the time. 
- The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .01. 
- The recall score for high-risk loans was 62% and 65% for low-risk loans. These scores indicate a relatively large number of false negatives,  so the model does poorly to identify high-risk loans.

#### Undersampling - ClusterCentroids
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/ClusterCentroids.png "Title")
- The Random Oversampling accuracy score was .5294. Meaning the `y_pred` model was accurate 52.94% of the time. 
- The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .01. 
- The recall score for high-risk loans was 61% and 45% for low-risk loans. These scores indicate a relatively large number of false negatives. The high-risk loans have fewer false negatives than the low-risk ones, but both do poorly to find positive samples. 
The undersampling ClusterCentroids algorithm, compared to the oversampling algorithms, does a worse job at identifying high-risk loans. 

#### Combination (Over and Under) Sampling - SMOTEENN
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/SMOTEENN.png "Title")
- The Random Oversampling accuracy score was .6376. Meaning the `y_pred` model was accurate 63.76% of the time. 
- The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .01. 
- The recall score for high-risk loans was 70% and 57% for low-risk loans. The score for high-risk loans indicates a relatively low number of false negatives, so the model does a fair job identifying high-risk loans. The low-risk loan score indicates a large number of false negatives, and the model does a poor job of identifying low-risk loans. 

#### Ensemble Learners - Balanced Random Forest Classifier
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/ForestClassifier.png "Title")
- The Random Oversampling accuracy score was .7878. Meaning the `y_pred` model was accurate 78.78% of the time. 
- The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .01. 
- The recall score for high-risk loans was 67% and 91% for low-risk loans. The high-risk loan scores indicate a relatively moderate number of false negatives, so the model does a fair job of identifying high-risk loans. The low-risk loan scores indicate a low number of false negatives, so the model does an excellent job of identifying low-risk loans. 

#### Ensemble Learners - Easy Ensemble AdaBoost Classifier
![Alt text](https://raw.githubusercontent.com/jmsuarez1997/Credit_Risk_Analysis/main/Resources/AdaBoostClassifier.png "Title")
- The Random Oversampling accuracy score was .9254. Meaning the `y_pred` model was accurate 92.54% of the time. The `EasyEnsembleClassifier` model has the highest accuracy score of all the tested models.
- The precision scores for this model favored the low-risk loans and had a perfect score of 1. However, the model did a poor job predicting the high-risk loans and had a score of .07. 
- The recall score for high-risk loans was 91% and 94% for low-risk loans. These scores indicate a relatively low number of false negatives for low-risk and high-risk loans. The model has an excellent recall rate for identifying high-risk and low-risk loans.

## Summary:

The AdaBoost Classifier model had the highest accuracy scores at 92.54%. The other model's accuracy scores stayed between 52.94%  and 78.78%. All our ML models had low precision scores for high-risk loans and perfect scores for low-risk loans. The AdaBoost Classifier also had the highest recall rates for high and low-risk loans at 91% and 94%. The Random Forest Classifier also had a high recall rate for low-risk loans at 91%. I recommend using the AdaBoost Classifier model to predict low and high-risk loans because this model has the highest accuracy and recall score. However, the model's precision score is low and unreliable for predicting a positive classification. The model may need a few more adjustments and more data to train and test to improve the precision score. 
