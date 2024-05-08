#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


dataset = pd.read_csv("E:\CTU-13-Dataset\8\capture20110816-3.binetflow")
print(dataset.to_string())
#dataset = dataset[dataset['Label'].str.contains('Botnet')] #Getting the bot net data
#print('Rows:',dataset.shape[0], 'Columns:', dataset.shape[1])
#columns = list(dataset.columns)
#print(columns)
#dataset.describe()


# In[32]:


dataset.sample(10)
#dataset['Label'].nunique()


# In[33]:


#Visualizing the Missing value
plt.figure(figsize=(16,5))
sns.set_style('whitegrid')
plt.title('% of Missing Values')
sns.barplot(x=dataset.isnull().mean().index, y=dataset.isnull().mean().values)
plt.xlabel('Columns')
plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.show()

# Removed all the columns with missing value > 30%
dataset = dataset.loc[:, dataset.isnull().mean() < 0.3] 


# In[34]:


dataset = dataset.astype({"Proto":'category',"Sport":'category',"Dport":'category',"State":'category','StartTime':'datetime64[s]'}) # Changing the datatype of the columns
dataset['duration'] = abs(dataset['StartTime'].dt.second) # getting duration from the columns 'LastTime' and 'StartTime'
dataset.drop(columns=['SrcAddr','DstAddr','StartTime'],inplace=True) #Dropping the column SrcAddr and DstAddr since they contain unique ip add


# In[36]:


print(dataset.shape)


# In[37]:


#Checking for null values 
print(dataset.isnull().sum())


# In[39]:


# #Checking for duplicates
print(dataset.duplicated().any())
print(dataset.duplicated())


# In[6]:


#Analyzing categorical valriable
def barchart(columns):
    plt.figure(figsize=(10,5))
    plt.title(f'{columns}')
    sns.countplot(x=dataset[f'{columns}'].value_counts().values)
    plt.xlabel(f'{columns}')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.show()
    
categorical_columns = dataset.select_dtypes(exclude=['int64', 'float64']).columns.values      
for column in categorical_columns:
    if column != 'Label':
        barchart(column)


# In[40]:


dataset = pd.get_dummies(dataset,columns=categorical_columns[:-1],drop_first=True)
X = dataset.loc[:, dataset.columns != 'Label']
y = dataset.loc[:, dataset.columns == 'Label']


# In[8]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=45)


# In[10]:


descision_tree_model = DecisionTreeClassifier()
descision_tree_model.fit(Xtrain,ytrain)
prediction = descision_tree_model.predict(Xtest)
print('Decision Accuracy Score:',round(accuracy_score(ytest,prediction)*100),'%')


# In[11]:


multinomial_naive_bayes = GaussianNB()
multinomial_naive_bayes.fit(Xtrain,ytrain)
prediction_naive = multinomial_naive_bayes.predict(Xtest)
print('Naive Bayes Accuracy Score:',round(accuracy_score(ytest,prediction_naive)*100),'%')


# In[12]:


Logistic_model = LogisticRegression(C=1000)
Logistic_model.fit(Xtrain,ytrain)
prediction_Logistic = Logistic_model.predict(Xtest)
print('Logistic Regression Accuracy Score:',round(accuracy_score(ytest,prediction_Logistic)*100),'%')


# In[13]:


Random_forest_model = RandomForestClassifier(class_weight='balanced')
Random_forest_model.fit(Xtrain,ytrain)
prediction_Random_forest_model = Random_forest_model.predict(Xtest)
print('Random Forest Accuracy Score:',round(accuracy_score(ytest,prediction_Random_forest_model)*100),'%')


# In[ ]:




