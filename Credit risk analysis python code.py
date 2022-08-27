#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset=pd.read_csv("credit_risk_dataset.csv")
dataset.head()


# In[3]:


dataset.isnull().sum()


# In[5]:


dataset['person_emp_length'].fillna(dataset['person_emp_length'].mean())


# In[6]:


dataset['loan_int_rate'].fillna(dataset['loan_int_rate'].mean())


# In[7]:


dataset.info()


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['person_home_ownership'] = le.fit_transform(dataset['person_home_ownership'])
dataset['loan_intent'] = le.fit_transform(dataset['loan_intent'])
dataset['loan_grade'] = le.fit_transform(dataset['loan_grade'])
dataset['cb_person_default_on_file'] = le.fit_transform(dataset['cb_person_default_on_file'])
dataset['person_emp_length'] = le.fit_transform(dataset['person_emp_length'])
dataset['loan_int_rate'] = le.fit_transform(dataset['loan_int_rate'])
dataset['loan_percent_income'] = le.fit_transform(dataset['loan_percent_income'])


# In[19]:


y = dataset['loan_status']
y.head()


# In[20]:


X = dataset.drop(['loan_status'], axis = 1)
X.head()


# In[21]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[26]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


# In[37]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[38]:


formula = 'loan_status ~ person_age+person_income+person_home_ownership+person_emp_length+loan_intent+loan_grade+loan_amnt+loan_int_rate+loan_percent_income+cb_person_default_on_file+cb_person_cred_hist_length'


# In[39]:


model = smf.glm(formula = formula, data=dataset,family=sm.families.Binomial())
result = model.fit()
print(result.summary())


# In[ ]:


#In the above summary we observe that person_home_ownership,person_emp_length,loan_intent,loan_grade,loan_amnt,loan_percent_income
#are statistically insignificant 


# In[44]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[45]:


y_pred = classifier.predict(X_test)


# In[46]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# In[ ]:




