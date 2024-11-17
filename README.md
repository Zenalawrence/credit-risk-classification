# credit-risk-classification

## Overview of the Analysis

The purpose of the model was to determine whether a loan is classified as healthy (class 0) or high-risk (class 1). This analysis served to develop a supervised machine learning model, Logistic regression to predict a dataset consisting of 77,500 entries. Logistic regression from **sklearn.linear_model** was used because it is a binary classifier that the characteristic of our data. A number of factors were taken into consideration for this analysis including:

- Loan size
- Interest rate
- Borrower income
- Debt-to-income ratio
- Number of accounts
- Derogatory marks
- Total debt

1. Firstly the loan_status data was stored into a target variable 'y' from the dataset.  The remaining columns listed above were stored into feature variable, 'x'.

2. From the 'sklearn.model_selection' module, the 'train_test_split' function was then used to split the data into training and testing subsets, 'x_train', 'y_train', 'x_test' and 'y_test'.  A random_state of 1 was applied ensuring the same data points were used for the sets allowing for consistent results.

3.  The logistic regression model then trained the data in 'x_train' and 'y_train' by using the 'fit' method of **sklearn**  with a random_state of 1.  This allows the model to learn the relationship between the features and the target labels.

4. the 'predict' method was then used on the 'x_test' to generate predictions based on the trained data.  The results was then stored in a 'pandas' DataFrame under "Predictions" and "Actual" Data.

5.  From the 'sklearn.metrics' module, the 'confusion_matrix' and 'accuracy_score' functions were used to test the accuracy of the logistic regression model on the dataset.  

6.  This accuracy results were then printed into an 'classification_report' from the sklearn.metrics module.


## Results

![confusion_matrix summary report.](xxx)

The performance of the logistic regression model was tested and stored in the classification report above.

**1. Accuracy Score:**
  - The accuracy score was found to be 0.99246 which is 99.2%.

**1. Precision:**

  - For healthy loans(class 0), the prediction was 100% which mean all predicted instances were correct.
  - For high-risk loans(class 1), the prediction was 84% correct meaning there were some false positives.

**2. Recall:** 

  - For healthy loans, the value 0.99 indicates that 99% of the actual instances of class 0 were correctly predicted.
  - For high-risk loans, 94% of the actual instances of class 1 were correctly identified.


## Summary

The model is good at  predicting healthy loans(class 0) with 100% accuracy but is not as strong at predicting high-risk loans as it has 84% accuracy prediction.  Improvements can be made using a more balanced dataset.  The two classes are currently imbalanced as healthy loans had 18765 dataset compard to the 619 dataset for high-risk loans.  Techniques like Random Over Sampling (RO), Random Under Sampling(RU), or synthetic data generation (SMOTE) can help improve performance on the minority class [^1].  The predictions may also be improved by tuning the hyperparametes for better overall performance for both classes.


## References

| Reference Number | link |
|----------------|-------------|
| [^1]: Imbalanced Data Techniques | https://www.turintech.ai/what-is-imbalanced-data-and-how-to-handle-it/ |

