#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pickle as pkl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[4]:


model = pkl.load(open('Titanic_model.pkl','rb'))
st.title('Titanic Survival Prediction App')


# In[30]:


def user_input_parameters():
    Pclass = st.sidebar.selectbox('Passenger Class (1=1st, 2=2nd, 3= 3rd)',[1,2,3])
    Age = st.sidebar.slider('Age',0,30,100)
    Sibsp = st.sidebar.number_input('Sibling/spouse Abroad', value= 0)
    Parch = st.sidebar.number_input('Parent/child Abroad', value= 0)
    Fare = st.sidebar.number_input('Fare paid', value= 35.0)
    Sex_male = st.sidebar.selectbox('Gender(Male =1, Female=0)',[0,1])
    Embarked_C= st.sidebar.selectbox('Embarked_C (Yes =1, No=0)',[0,1])
    Embarked_Q= st.sidebar.selectbox('Embarked_Q (Yes =1, No=0)',[0,1])
    Embarked_S= st.sidebar.selectbox('Embarked_S (Yes =1, No=0)',[0,1])

    data = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': Sibsp,
        'Parch': Parch,
        'Fare': Fare,
        'Sex_male': Sex_male,
        'Embarked_C': Embarked_C,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S

    }
    features= pd.DataFrame(data, index=[0])
    return features


# In[31]:


df = user_input_parameters()

st.subheader('User Input Parameters')
st.write(df)


# In[32]:


pred = model.predict(df)
pred_prob = model.predict_proba(df)


# In[33]:


st.subheader('Predicted Survival')
st.write('WoW (Got another Day to Live' if pred[0]== 1 else 'Haa! Yumraj spotted. Better luck next time!')


# In[35]:


st.subheader('Prediction Probability')
st.write(pred_prob)


# In[ ]:




