#!/usr/bin/env python
# coding: utf-8

# ## Iris Classification Machine learning Project
# 
# 
# 

# In[ ]:


import numpy as np


# In[ ]:


#importing dataset of IRIS
from sklearn.datasets import load_iris


# In[ ]:


#saving dataset of IRIS in iris_dataset
iris_dataset = load_iris()


# In[ ]:


print(iris_dataset.data)


# In[ ]:


#printing the column names(features)
print (iris_dataset.feature_names)


# In[ ]:


#printing result type : 0 for setosa, 1 for versicolor, 2 for virginica
print(iris_dataset.target)


# In[ ]:


print(iris_dataset.target_names)


# In[ ]:


#storing data and target
X=iris_dataset.data
y=iris_dataset.target


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


X,y = shuffle(X,y,random_state = 0)


# In[ ]:


print(X)


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


#importing class
from sklearn.neighbors import KNeighborsClassifier


# 
# **For nearest one**

# In[ ]:


#instantiation
knn = KNeighborsClassifier(n_neighbors = 1)


# In[ ]:


#model training
knn.fit(X,y)


# In[ ]:


#model testing
#0 for setosa, 1 for versicolor, 2 for virginica
knn.predict([[3,5,4,2]])


# In[ ]:


X_new = ([[3,5,4,2],[1,2,3,4]])
knn.predict(X_new)


# **best among 5**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 5)


# In[ ]:


knn.fit(X,y)


# In[ ]:


knn.predict(X_new)


# **Using Logistic Regression**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42 )


# In[ ]:


y_train.shape


# In[ ]:


#imporitng class
from sklearn.linear_model import LogisticRegression


# In[ ]:


#instantiation
lr = LogisticRegression()


# In[ ]:


#model training
lr.fit(X,y)


# In[ ]:


#model testing
y_pred = lr.predict(X_test)
print(y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)


# In[ ]:


print(acc)

