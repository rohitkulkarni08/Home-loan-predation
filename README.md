# Loan Default Prediction Project

## Overview
This project focuses on predicting loan defaults using various machine learning models. The goal is to help financial institutions identify potential defaulters before approving loans. The project involves exploratory data analysis (EDA) to understand the dataset's characteristics and modeling to predict loan defaults.

## Dataset
The [Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) used in this project contains information about borrowers, including their credit history, loan amount, income level, and other relevant features that influence the likelihood of defaulting on a loan.

## Key Features:

1. LoanID
2. Age
3. Income
4. LoanAmount
5. CreditScore
6. MonthsEmployed
7. NumCreditLines
8. InterestRate
9. LoanTerm
10. DTIRatio
11. Education
12. EmploymentType
13. MaritalStatus
14. HasMortgage
15. HasDependents
16. LoanPurpose
17. HasCoSigner
    
The main variable of interest is **Default**

## Models Used
- Logistic Regression
- Random Forest
- Decision Tree

## Requirements
- Python 3.8 or above
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn


## Results
The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1 score. The best-performing model is selected based on these metrics for predicting loan defaults.

After initially running the model, we got the following results:

| Model               | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|---------------------|----------|--------------|----------|---------------|
| Logistic Regression | 0.885020 | 0.031507     | 0.059939 | 0.747049      |
| Random Forest       | 0.881981 | 0.058974     | 0.104162 | 0.668864      |
| Decision Tree       | 0.815655 | 0.194695     | 0.197271 | 0.546053      |


Recognizing that a lot of the evaluation metrics had low scores, we used a random forest regressor to help with the feature selection and remove unnecessary features from the model.

| Model               | Accuracy | Recall Score | F1 Score | ROC AUC Score |
|---------------------|----------|--------------|----------|---------------|
| Logistic Regression | 0.672181 | 0.655581     | 0.317561 | 0.725739      |
| Random Forest       | 0.881855 | 0.044836     | 0.081140 | 0.655326      |
| Decision Tree       | 0.816172 | 0.200619     | 0.202514 | 0.548918      |

By selecting important features using RandomForestRegressor, we can observe some changes in the model performance metrics:
1. Accuracy:
   
   a. Logistic Regression had a decrease in accuracy from 0.885020 to 0.672181, indicating that the model is now correctly predicting a lower percentage of the total observations
   
   b. Random Forest had a marginal decrease from 0.881981 to 0.881855. The model's overall prediction accuracy remains relatively unchanged
   
   c. Decision Tree had a marginal increase from 0.815655 to 0.816172. The overall prediction accuracy of the model remains relatively unchanged
   
2. Recall:
  
   a. Logistic Regression had a significant increase from 0.031507 to 0.655581. This means that the model is now much better at identifying true positives

   b. Random Forest had a decrease from 0.058974 to 0.044836. The model is now slightly worse at identifying true positives

   c. Decision Tree had a slight increase from 0.194695 to 0.200619. The model is now slightly better at identifying true positives

3. F1 score:
  
   a. Logistic Regression had a steep rise from 0.059939 to 0.317561, suggesting that the balance between precision and recall has improved, resulting in a higher F1 score

   b. Random Forest and Decision Tree had a slight increase indicating a slight improvement in the balance between precision and recall

4. ROC AUC Score:
  
   a. Logistic Regression and Random Forest had a slight decrease indicating a slight decrease in the model's ability to distinguish between the positive and negative classes

   b. Decision Tree had a slight increasing from 0.546053 to 0.548918, suggesting a slight improvement in the model's ability to distinguish between the positive and negative classes

## Conclusion

The feature selection using RandomForestRegressor has led to a significant improvement in the recall score for Logistic Regression, indicating that the model is now better at identifying the positive class.

However, this improvement comes at the cost of reduced accuracy. The Random Forest model saw a slight decrease in performance across all metrics, while the Decision Tree model saw marginal improvements. Overall, the changes in model performance after feature selection suggest that the selected features have a more significant impact on the Logistic Regression model's ability to identify the positive class.

