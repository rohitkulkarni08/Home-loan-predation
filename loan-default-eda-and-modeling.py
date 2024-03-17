#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,f1_score,roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv('Loan_default.csv')
df.head()


# In[3]:


print("Data set size:")
df.shape


# In[4]:


df.info()


# In[5]:


df.columns


# ## Exploratory Data Analysis

# In[6]:


df.describe()


# In[7]:


print("Missing values in each column:")
df.isnull().sum()


# In[8]:


print("Observing Default Values:")
df['Default'].value_counts()


# In[9]:


plt.figure(figsize=(4, 4))
output_counts = df['Default'].value_counts()
plt.pie(output_counts, labels=output_counts.index, autopct='%1.1f%%', startangle=140)

plt.axis('equal')
plt.title('Distribution of Default Status \n')
plt.ylabel('')

plt.show()


# In[10]:


print("Observing Age Values:")
df['Age'].describe()


# In[11]:


def age_group(age):
    if age < 18:
        return '<18'
    elif age >= 18 and age <=30:
        return 'Between 18 and 30'
    elif age >= 31 and age <=40:
        return 'Between 31 and 40'
    elif age >= 41 and age <=50:
        return 'Between 41 and 50'
    elif age >= 51 and age <=60:
        return 'Between 51 and 60'
    elif age >= 61 and age <=70:
        return 'Between 61 and 70'
    else:
        return 'Greater than 70'

print("Observing Age Values by buckets:", '\n')
df['age_buckets'] = df['Age'].apply(age_group)
print(df['age_buckets'].value_counts())


# In[12]:


print("Countplot for numerical features:")
plt.figure(figsize=(24,25))
countplot_features = ['age_buckets','Education','HasDependents','MaritalStatus','LoanPurpose','NumCreditLines','LoanTerm','HasCoSigner']
for i, column in enumerate(countplot_features):
    plt.subplot(4,2, i + 1)
    sns.countplot(x=df[column], width=0.4)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')


# #### Observation:
# 
# There are more users in the age bucket "Between 18 and 30". For remaining columns, the values are almost equally distributed.

# In[13]:


print("Boxplot for numerical features:")

boxplot_features = ['Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']
plt.figure(figsize=(18, 22))
for i, column in enumerate(boxplot_features):
    plt.subplot(4, 2, i + 1)
    sns.boxplot(x=df[column], width=0.4)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')


# #### Observation:
# 
# There aren't any noticable outliers for any of the features.

# ## Feature Engineering

# In[14]:


bins = [0,60000,120000,180000, float('inf')]
labels = ['Low','Average','High','Very High']

df['LoanAmount_Bins'] = pd.cut(df['LoanAmount'], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(6,3))
sns.countplot(x = 'LoanAmount_Bins', hue = 'Default', data = df)
plt.title('Observing the Loan Amount buckets based on the Exited column')
plt.show()


# #### Observation:
# 
# People with Average, High, and Cery High loan amounts have defaulted as compared to lower amounts

# In[15]:


bins = [0,669,739,850]
labels = ['Low','Medium','High']

df['CreditScoreGroup'] = pd.cut(df['CreditScore'], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(6,3))
sns.countplot(x = 'CreditScoreGroup', hue = 'Default', data = df)
plt.title('Observing the Credit Score buckets based on the Default column')
plt.show()


# #### Observation:
# 
# People with Low Credit Scores have defaulted their loans a lot more in comparison to people with Medium and High Credit Scores

# In[16]:


corr_matrix = df.corr()
plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, vmax=.8, square=True,annot=True);


# #### Observation:
# 
# The heatmap shows correlations between variables in a home loan dataset:
# 
# 1. **Loan Amount and Income**: Moderately positive correlation suggests higher incomes are associated with larger loan amounts.
# 2. **Default and Months Employed**: Negative correlation implies that longer employment is associated with a lower risk of default.
# 3. **Default and Age**: Negative correlation indicates younger applicants may be at a higher risk of default.
# 4. **Interest Rate and Credit Score**: Negative correlation suggests that better credit scores may lead to lower interest rates.
# 5. **Loan Term and Number of Credit Lines**: Slight positive correlation could mean longer loan terms are associated with having more credit lines.

# In[17]:


target_correlations = corr_matrix['Default']
print(target_correlations)


# #### Observation:
# 
# 1. **Age**: A moderate negative correlation (-0.17) with defaulting suggests younger applicants are more likely to default.
# 2. **Income** and **Months Employed**: Both have a small negative correlation with defaulting (-0.10 and -0.097, respectively), indicating that higher income and longer employment might reduce the likelihood of default.
# 3. **Loan Amount**: A small positive correlation (0.087) with defaulting, suggesting higher loan amounts could be slightly associated with an increased risk of default.
# 4. **Credit Score**: A small negative correlation (-0.034) with defaulting, implying better credit scores are associated with a lower risk of default.
# 5. **NumCreditLines**: A very small positive correlation (0.028) with defaulting, indicating a slight increase in default risk with more credit lines.
# 6. **Interest Rate**: A moderate positive correlation (0.13) with defaulting, suggesting higher interest rates may be associated with higher default rates.
# 7. **Loan Term** and **DTIRatio**: Very small positive correlations with defaulting (0.0005 and 0.019, respectively), indicating negligible effects on the likelihood of default.

# In[18]:


df


# ## Modeling

# In[19]:


cat_columns = ['Education','EmploymentType','MaritalStatus','HasMortgage','HasDependents','LoanPurpose','HasCoSigner','age_buckets','LoanAmount_Bins','CreditScoreGroup']
print("Observing the categorical column disribution before encoding: \n")

print("Observing the categorical column disribution before encoding: \n")
for columns in cat_columns:
    print(columns, '\n')
    print(df[columns].value_counts(),'\n')


# In[20]:


encoder = LabelEncoder()

for columns in cat_columns:
    df[columns] = encoder.fit_transform(df[columns])

print("Observing the categorical column disribution after encoding: \n")    
for columns in cat_columns:
    print(columns, '\n')
    print(df[columns].value_counts(),'\n')


# In[21]:


col_drop = ['LoanID','Default']
X = df.drop(col_drop, axis=1)
y = df['Default']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)


# In[22]:


scaling_columns = ['Age','Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']

scaler = StandardScaler()
scaler.fit(x_train[scaling_columns])

x_train[scaling_columns] = scaler.transform(x_train[scaling_columns])
x_test[scaling_columns] = scaler.transform(x_test[scaling_columns])


# In[23]:


models = {
    'Logistic Regression': LogisticRegression(random_state = 42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

results_df = pd.DataFrame(columns=['Model','Accuracy','Recall Score','F1 Score','ROC AUC Score'])

lb = LabelBinarizer()
lb.fit(y_train)

for name, model in models.items():
    print(f"Model: {name}")
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred),'\n')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred),'\n')
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy} \n")
    
    recall = recall_score(y_test, y_pred, pos_label=1)
    print(f"Recall Score: {recall}")

    f1 = f1_score(lb.transform(y_test), lb.transform(y_pred), pos_label=1)
    print(f"F1 Score: {f1}")
    
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(lb.transform(y_test), model.predict_proba(x_test)[:, 1])
        print(f"ROC AUC Score: {roc_auc}")
    else:
        roc_auc = None
    
    results_df = results_df.append({'Model': name, 'Accuracy': accuracy, 'Recall Score': recall, 'F1 Score': f1, 'ROC AUC Score': roc_auc}, ignore_index=True)
    
    print("-" * 50,'\n')


# In[24]:


results_df


# #### Observations:
# 
# 1. **Logistic Regression** has the highest accuracy (**88.5%**) and the highest ROC AUC Score (**0.747**), suggesting it is quite good at classifying but not as sensitive in detecting the positive class (**low recall**).
# 2. **Random Forest** has slightly lower accuracy (**88.2%**) and ROC AUC Score (**0.669**) compared to Logistic Regression. Its recall and F1 score are higher than those for Logistic Regression, which means it's better at identifying the positive class but overall it makes more mistakes.
# 3. **Decision Tree** has the lowest accuracy (**81.5%**) and ROC AUC Score (**0.546**), which indicates it is not as good at distinguishing between classes as the other models. However, it has the highest recall (**19.49%**), suggesting it's better at identifying the positive class than the other models, but this comes at the cost of making more overall prediction errors.
# 
# Let's try and improve the model...

# ## Using RandomForest to get important features

# In[25]:


x2 = df.drop(['LoanID','Default'], axis=1)
y2 = df['Default']

rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regressor.fit(x2, y2)
feature_importances = rf_regressor.feature_importances_


importance_df = pd.DataFrame({'Feature': x2.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  
plt.show()


# In[26]:


importance_df


# #### Observation:
# 
# There are a lot of features with low importance. Let's select features with an importance greater than 0.03, which seems like a reasonable cutoff based on the given importances. 

# In[27]:


selected_features = ['Income', 'InterestRate', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'DTIRatio', 'Age', 'LoanPurpose', 'LoanTerm']

selected_df = df[selected_features]

x_train2, x_test2, y_train2, y_test2 = train_test_split(selected_df,df['Default'],test_size=0.25,random_state=42)

print('x_train2:',x_train2.shape)
print('y_train2:',y_train2.shape)
print('x_test2:',x_test2.shape)
print('y_test2:',y_test2.shape)


# In[28]:


scaling_columns_2 = ['Age','Income','LoanAmount','CreditScore','MonthsEmployed','InterestRate','LoanTerm','DTIRatio']

scaler = StandardScaler()
scaler.fit(x_train[scaling_columns_2])

x_train2[scaling_columns_2] = scaler.transform(x_train2[scaling_columns_2])
x_test2[scaling_columns_2] = scaler.transform(x_test2[scaling_columns_2])


# In[29]:


models2 = {
    'Logistic Regression': LogisticRegression(random_state = 42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

results_df2 = pd.DataFrame(columns=['Model','Accuracy','Recall Score','F1 Score','ROC AUC Score'])

lb = LabelBinarizer()
lb.fit(y_train2)

for name, model in models2.items():
    print(f"Model: {name}")
    
    model.fit(x_train2, y_train2)
    y_pred2 = model.predict(x_test2)
    print(classification_report(y_test2, y_pred2),'\n')
    print("Confusion Matrix:")
    print(confusion_matrix(y_test2, y_pred2),'\n')
    
    accuracy_2 = accuracy_score(y_test2, y_pred2)
    print(f"Accuracy Score: {accuracy_2} \n")
    
    recall_2 = recall_score(y_test2, y_pred2, pos_label=1)
    print(f"Recall Score: {recall_2}")

    f1_2 = f1_score(lb.transform(y_test2), lb.transform(y_pred2), pos_label=1)
    print(f"F1 Score: {f1_2}")
    
    if hasattr(model, "predict_proba"):
        roc_auc_2 = roc_auc_score(lb.transform(y_test2), model.predict_proba(x_test2)[:, 1])
        print(f"ROC AUC Score: {roc_auc_2}")
    else:
        roc_auc_2 = None
    
    results_df2 = results_df2.append({'Model': name, 'Accuracy': accuracy_2, 'Recall Score': recall_2, 'F1 Score': f1_2, 'ROC AUC Score': roc_auc_2}, ignore_index=True)
    
    print("-" * 50,'\n')


# In[30]:


results_df2


# #### Observation:
# 
# By selecting important features using RandomForestRegressor, we can observe some changes in the model performance metrics:
# 
# 1. Accuracy:
#    - **Logistic Regression** had a decrease in accuracy from 0.885020 to 0.672181, indicating that the model is now correctly predicting a lower percentage of the total observations
#    - **Random Forest** had a marginal decrease from 0.881981 to 0.881855. The model's overall prediction accuracy remains relatively unchanged
#    - **Decision Tree** had a marginal increase from 0.815655 to 0.816172. The overall prediction accuracy of the model remains relatively unchanged
# 
# 2. Recall:
#    - **Logistic Regression** had a significant increase from 0.031507 to 0.655581. This means that the model is now much better at identifying true positives
#    - **Random Forest** had a decrease  from 0.058974 to 0.044836. The model is now slightly worse at identifying true positives
#    - **Decision Tree** had a slight increase from 0.194695 to 0.200619. The model is now slightly better at identifying true positives
# 
# 3. F1 score:
#    - **Logistic Regression** had a steep rise  from 0.059939 to 0.317561, suggesting that the balance between precision and recall has improved, resulting in a higher F1 score
#    - **Random Forest** and **Decision Tree** had a slight increase indicating a slight improvement in the balance between precision and recall
# 
# 4. ROC AUC Score:
#    - **Logistic Regression** and **Random Forest** had a slight decrease indicating a slight decrease in the model's ability to distinguish between the positive and negative classes
#    - **Decision Tree** had a slight increasing from 0.546053 to 0.548918, suggesting a slight improvement in the model's ability to distinguish between the positive and negative classes
# 
# #### Overall:
# 
# The feature selection using RandomForestRegressor has led to a significant improvement in the recall score for Logistic Regression, indicating that the model is now better at identifying the positive class. 
# 
# However, this improvement comes at the cost of reduced accuracy. The Random Forest model saw a slight decrease in performance across all metrics, while the Decision Tree model saw marginal improvements. Overall, the changes in model performance after feature selection suggest that the selected features have a more significant impact on the Logistic Regression model's ability to identify the positive class.

# In[ ]:




