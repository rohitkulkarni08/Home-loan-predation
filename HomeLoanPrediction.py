#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data handling
import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , StandardScaler

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# # Load Data

# In[2]:


df = pd.read_csv('Loan_default.csv')
df.head()


# In[3]:


print("Data set size:")
print(f"{df.shape[0]} rows")
print(f"{df.shape[1]} columns")


# In[4]:


df.columns


# In[5]:


df.info()


# # EDA

# In[6]:


print("Duplicate rows:")
print(df.duplicated().sum())


# In[7]:


print("Null rows:")
df.isnull().sum()


# In[8]:


print("Observing Default Values:")
print(df['Default'].value_counts(),'\n')
print("Normalized Default Values:")
print(df['Default'].value_counts(normalize=True)*100,'\n')
df['Default'].value_counts(normalize=True).plot.bar(title = 'Default Status')


# In[9]:


print("Observing Age Values:")
df['Age'].describe()


# In[10]:


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
print(df['age_buckets'].value_counts(normalize = True)*100)
(df['age_buckets'].value_counts(normalize = True)*100).plot.bar(title = 'Age Groups')


# In[11]:


print(df['Education'].value_counts())
df['Education'].value_counts().plot.bar(title = 'Education')


# In[12]:


print(df['MaritalStatus'].value_counts())
(df['MaritalStatus'].value_counts(normalize=True)*100).plot.bar(title = 'Marital Status')


# In[13]:


print(df['HasDependents'].value_counts())
(df['HasDependents'].value_counts(normalize=True)*100).plot.bar(title = 'Dependents')


# In[14]:


print(df['LoanPurpose'].value_counts())
(df['LoanPurpose'].value_counts(normalize=True)*100).plot.bar(title = 'Purpose')


# In[15]:


plt.figure(figsize=(20, 22))
for i, column in enumerate(['Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']):
    plt.subplot(4, 2, i + 1)
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')


# In[16]:


plt.figure(figsize=(18, 22))
for i, column in enumerate(['Income','LoanAmount','CreditScore','MonthsEmployed','NumCreditLines','InterestRate','LoanTerm','DTIRatio']):
    plt.subplot(4, 2, i + 1)
    sns.boxplot(x=df[column], color='skyblue', width=0.4)
    plt.title(column)
    plt.xlabel('Value')
    plt.ylabel('Frequency')


# In[17]:


df.boxplot(column='Income',by="Education" )
plt.suptitle("Income and Education Boxplot")
plt.show()


# In[18]:


df['LoanAmount'].describe()


# In[19]:


df["LoanAmount_Bins"]=pd.cut(df["LoanAmount"],[0,60000,120000,180000],labels=['Low','Average','High'])
print(df["LoanAmount_Bins"].value_counts())


# In[20]:


print(pd.crosstab(df["LoanAmount_Bins"],df["Default"]))
LoanAmount=pd.crosstab(df["LoanAmount_Bins"],df["Default"])
LoanAmount.div(LoanAmount.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel("LoanAmount")
plt.ylabel("Percentage")
plt.show()


# In[21]:


corr_matrix = df.corr()
plt.subplots(figsize=(10, 12))
sns.heatmap(corr_matrix, vmax=.8, square=True,annot=True);


# # Pre-Processing

# In[22]:


encoder = LabelEncoder()

df['Education'] = encoder.fit_transform(df['Education'])
df['EmploymentType'] = encoder.fit_transform(df['EmploymentType'])
df['MaritalStatus'] = encoder.fit_transform(df['MaritalStatus'])
df['HasMortgage'] = encoder.fit_transform(df['HasMortgage'])
df['HasDependents'] = encoder.fit_transform(df['HasDependents'])
df['LoanPurpose'] = encoder.fit_transform(df['LoanPurpose'])
df['HasCoSigner'] = encoder.fit_transform(df['HasCoSigner'])

scaler = StandardScaler()
df['Income'] = scaler.fit_transform(df[['Income']])
df['LoanAmount'] = scaler.fit_transform(df[['LoanAmount']])
df['CreditScore'] = scaler.fit_transform(df[['CreditScore']])


# In[23]:


df.drop(['LoanID','age_buckets','LoanAmount_Bins'],axis=1,inplace = True)


# In[24]:


x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=['Default']),df['Default'],test_size=0.25,random_state=42)

print('x_train_1:',x_train.shape)
print('y_train_1:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)


# In[25]:


y_train.value_counts()


# # Models

# In[26]:


#Logistic Regression
log_reg_model = LogisticRegression(random_state = 42, max_iter=1000)
log_reg_model.fit(x_train,y_train)

y_pred = log_reg_model.predict(x_test)
print(classification_report(y_test,y_pred),'\n')

score_log_reg = accuracy_score(y_test,y_pred)*100
print(score_log_reg)


# In[27]:


#Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(x_train,y_train)
y_test_tree = tree_model.predict(x_test)
score_tree = accuracy_score(y_test_tree,y_test)*100 
score_tree


# In[28]:


#Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(x_train,y_train)
y_test_rf = rf_model.predict(x_test)
score_rf = accuracy_score(y_test_rf,y_test)*100 
score_rf


# ## Using RandomForest to get important features

# In[29]:


x = df.drop(['Default'], axis=1)
y = df['Default']

rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
rf_regressor.fit(x, y)
feature_importances = rf_regressor.feature_importances_


importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  
plt.show()


# In[31]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(df[['Income','LoanAmount','InterestRate','CreditScore','MonthsEmployed','DTIRatio','Age','LoanTerm','NumCreditLines','EmploymentType','Education']],df['Default'],test_size=0.25,random_state=42)

print('x_train2:',x_train2.shape)
print('y_train2:',y_train2.shape)
print('x_test2:',x_test2.shape)
print('y_test2:',y_test2.shape)


# In[32]:


#Logistic Regression-2
model_log_reg_2 = LogisticRegression(random_state = 42, max_iter=1000)
model_log_reg_2.fit(x_train2,y_train2)

y_pred2 = model_log_reg_2.predict(x_test2)
print(classification_report(y_test2,y_pred2),'\n')

score_log_reg_2 = accuracy_score(y_test2,y_pred2)*100
print(score_log_reg_2)


# In[33]:


#Decision Tree-2
tree_model_2 = DecisionTreeClassifier(random_state=42)
tree_model_2.fit(x_train2,y_train2)
y_test_tree_2 = tree_model_2.predict(x_test2)
score_tree_2 = accuracy_score(y_test_tree_2,y_test2)*100 
score_tree_2


# In[34]:


#Random Forest-2
rf_model_2 = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model_2.fit(x_train2,y_train2)
y_test_rf_2 = rf_model_2.predict(x_test2)
score_rf_2 = accuracy_score(y_test_rf_2,y_test2)*100 
score_rf_2


# In[ ]:




