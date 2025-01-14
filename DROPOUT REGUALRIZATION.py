#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
#header none is used kyuki it doenst have any col topic or head
df = pd.read_csv(r"C:\Users\DELL\Downloads\sonar_dataset.csv", header=None)
df.sample(5)



# In[26]:


df.shape


# In[27]:


df.isna().sum()


# In[28]:


df.columns


# In[29]:


df[60].value_counts()


# In[30]:


X = df.drop(60, axis=1)
y = df[60]
y.head()


# In[31]:


y = pd.get_dummies(y, drop_first=True)
y.sample(5) # R --> 1 and M --> 0


# In[32]:


y.value_counts()


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[34]:


X_train.head()


# In[35]:


X_test.head()


# In[36]:


#making deep leaning model
import tensorflow as tf
from tensorflow import keras


# In[37]:


#Dense(60): This creates a fully connected (dense) layer with 60 neurons.
#input_dim=60: This indicates that the input data has 60 features.
#activation='relu': The ReLU (Rectified Linear Unit) activation function is used,
    #which allows the model to learn complex patterns by introducing non-linearity.*/
  #  batch_size=8: The training data is split into batches of 8 samples,
        #and the model is updated after each batch, which speeds up training.


# In[38]:


model = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=8)


# In[39]:


model.evaluate(X_test, y_test)


# In[40]:


y_pred = model.predict(X_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1 .reshape(-1): This reshapes the predicted values into a 1D array. 
#The reshape(-1) ensures that the prediction results are flattened into a single column, which is helpful for further processing
y_pred = np.round(y_pred)
print(y_pred[:10])


# In[41]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))


# In[42]:


modeld = keras.Sequential([
    keras.layers.Dense(60, input_dim=60, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

modeld.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modeld.fit(X_train, y_train, epochs=100, batch_size=8)
 #put a dropout layer after hidden layer


# In[43]:


modeld.evaluate(X_test, y_test)


# In[44]:


y_pred = modeld.predict(X_test).reshape(-1)
print(y_pred[:10])

# round the values to nearest integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])


# In[45]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test, y_pred))


# In[ ]:




