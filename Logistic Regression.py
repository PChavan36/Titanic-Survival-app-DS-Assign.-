#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Task 1. Data Exploration:

# In[4]:


df_train= pd.read_csv('Titanic_train.csv')
df_test = pd.read_csv('Titanic_test.csv')


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


df_train.info()


# In[8]:


df_train.dtypes


# In[9]:


df_train.describe()


# In[27]:


sns.set_theme(style ='whitegrid')
plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
sns.histplot(df_train['Age'].dropna(), bins= 30, color = 'skyblue', kde= True)
plt.title('Distribution of passengers ages')
plt.xlabel('Age')

plt.subplot(1,2,2)
sns.boxplot(df_train['Fare'], color = 'lightgreen')
plt.title('Box plot of passenger fares')
plt.xlabel('Fare')


# In[30]:


plt.figure(figsize = (6,4))
sns.barplot(x='Sex', y= 'Survived', data = df_train, palette= 'Set2')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Probability')
plt.show()


# ### Task 2. Data Preprocessing:

# In[11]:


df_train.duplicated().sum()


# In[12]:


df_test.duplicated().sum()


# In[13]:


df_train.isnull().sum()


# In[14]:


df_test.isnull().sum()


# In[18]:


for i in df_train.isnull().sum():
    print(round(i/len(df_train)*100,2),'%')


# In[19]:


for i in df_test.isnull().sum():
    print(round(i/len(df_test)*100,2),'%')


# In[37]:


df_train_clean = df_train.drop(columns = ['PassengerId','Ticket', 'Cabin','Name'])
df_train_clean.head()


# In[38]:


df_test_clean = df_test.drop(columns = ['PassengerId','Ticket', 'Cabin', 'Name'])
df_test_clean.head()


# In[40]:


df_train_clean.fillna(df_train['Age'].median(),inplace = True)
df_test_clean.fillna(df_train['Age'].median(),inplace = True)


# In[41]:


df_train_clean.fillna(df_train['Embarked'].mode(),inplace = True)


# In[42]:


df1 = pd.get_dummies(df_train_clean, columns = ['Sex','Embarked'], drop_first = True)
df2 = pd.get_dummies(df_test_clean, columns = ['Sex','Embarked'], drop_first = True)


# In[86]:


df1.head()


# In[44]:


df1.isnull().sum()


# In[45]:


df2.isnull().sum()


# ### Task 3: Model Building

# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[50]:


features = df1.drop(columns = 'Survived')
target = df1['Survived']


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(features, target, train_size = 0.75, random_state = 60)


# In[54]:


log_model = LogisticRegression(random_state = 60)
log_model.fit(x_train, y_train)


# ### Task 4. Model Evaluation:

# In[61]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# In[56]:


y_pred = log_model.predict(x_test)


# In[57]:


y_prob = log_model.predict_proba(x_test)[:,1]


# In[64]:


print("--- Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.2%}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.2%}")


# In[65]:


fpr, tpr, threshold = roc_curve(y_test, y_prob)


# In[67]:


plt.figure(figsize = (8,5))
plt.plot(fpr, tpr, color = 'blue', linewidth = 2, label = 'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2%})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing (AUC = 0.50)')

plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ### Task 5. Interpretation:

# In[68]:


features_name = x_train.columns
coefficients = log_model.coef_[0]


# In[70]:


coef_df = pd.DataFrame({'Features': features_name, 'Coefficients': coefficients})


# In[71]:


coef_df.head()


# In[75]:


coef_df.sort_values(by = 'Coefficients', ascending= False)


# In[78]:


plt.figure(figsize = (8,4))
sns.barplot(x='Coefficients', y = 'Features', data = coef_df, palette= 'coolwarm')

plt.axvline(x=0, color= 'red', linestyle = '--')
plt.title('Feature Significance in Predicting Titanic Survival')
plt.xlabel('Coefficient (Impact on Survival Probability)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# ### Task 6. Deployment with Streamlit:

# In[81]:


import streamlit as st
import pickle as pkl


# In[85]:


file = 'Titanic_model.pkl'
pkl.dump(log_model, open(file,'wb'))


# In[ ]:




1. What is the difference between precision and recall?
Precision and recall are evaluation metrics that measure a classification model's performance from two different mathematical perspectives. Precision measures the accuracy of positive predictions: out of all the passengers the model predicted would survive, what percentage actually survived? It focuses on minimizing false alarms (False Positives). Recall measures the model's sensitivity: out of all the passengers who actually survived the Titanic, what percentage did the model successfully identify? It focuses on minimizing missed detections (False Negatives).

2. What is cross-validation, and why is it important in binary classification?
Cross-validation is a statistical resampling procedure used to evaluate how well a machine learning model generalizes to independent, unseen data. Instead of performing a single train/test split, it divides the dataset into multiple subsets (folds). The model is trained on several folds and tested on the remaining fold, with this process rotating until every fold has served as the test set. It is important in binary classification because it ensures the model's performance is stable across the entire dataset rather than relying on a potentially lucky or biased single split. Using stratified cross-validation further guarantees that the ratio of both classes (e.g., survived vs. died) remains perfectly proportionally represented during every test.
# In[ ]:




