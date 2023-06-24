#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:





# # Data Importing and Cleaning

# In[60]:


df = pd.read_csv("comments.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull()


# In[5]:


df.duplicated()


# In[61]:


df.isna().sum()


# In[62]:


mode_val = df['Comment'].mode()[0]

df['Comment'].fillna(mode_val, inplace = True)


# In[63]:


df.isna().sum()


# In[64]:


df1 = pd.read_csv('videos-stats.csv')
df1.head()


# In[65]:


#df1.rename(columns = {'Likes', 'Likes1'})
df1.rename(columns={'Likes': 'Likes1'}, inplace=True)


# In[66]:


merg = df1.merge(df, on='Video ID')


# In[67]:


merg


# In[68]:


df1 = merg.drop(['Unnamed: 0_y', 'Unnamed: 0_x'], axis = 1)


# In[69]:


df1


# In[15]:


df1.info()


# In[16]:


df1.describe()


# In[70]:


df1.isnull().sum()


# In[71]:


# df1.columns = df1.columns.str.strip()  # Remove leading/trailing whitespaces from column names

df1['Comments'].fillna(df1['Comments'].mode()[0], inplace=True)


# In[72]:


df1['Views'].fillna(df1['Views'].mode()[0], inplace=True)


# In[73]:


df1['Likes1'].fillna(df1['Likes1'].mode()[0], inplace=True)


# In[74]:


df1.isna().sum()


# In[ ]:





# # Data Visualization

# In[25]:


df.Likes.value_counts()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

likes_counts = df['Likes'].value_counts().head(200)  # Get top 200 counts
sns.countplot(data=df, x='Likes', order=likes_counts.index)  # Use order parameter to specify the order of bars
plt.show()


# In[27]:


sns.countplot(data=df,x='Sentiment')
plt.show()


# In[31]:


df1.columns.values


# In[33]:


import matplotlib.pyplot as plt

plt.scatter(df1['Views'], df1['Likes'])
plt.xlabel('Views')
plt.ylabel('Likes')
plt.title('Views vs Likes')
plt.show()


# In[35]:


avg_likes_by_keyword = df1.groupby('Keyword')['Likes'].mean()
plt.bar(avg_likes_by_keyword.index, avg_likes_by_keyword.values)
plt.xlabel('Keyword')
plt.ylabel('Average Likes')
plt.title('Average Likes by Keyword')
plt.xticks(rotation=90)
plt.show()


# In[38]:


sentiment_counts = df1['Sentiment'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.show()


# In[39]:


df1['Published Month'] = pd.to_datetime(df1['Published At']).dt.month
avg_likes_by_month = df1.groupby('Published Month')['Likes'].mean()
plt.bar(avg_likes_by_month.index, avg_likes_by_month.values)
plt.xlabel('Month')
plt.ylabel('Average Likes')
plt.title('Average Likes by Month')
plt.show()


# In[ ]:





# In[ ]:





# # modeling

# # linear regression

# In[75]:


x = df1["Likes"]
x.head()


# In[76]:


y = df1["Sentiment"]
y.head()


# In[42]:


# X_ = df1[['Title', 'Video ID', 'Published At', 'Keyword', 'Comments', 'Views']]
X_ = df1[['Comments', 'Views']]
y_ = df1['Likes']


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state = 0)
model = LinearRegression()
model.fit(x_train, y_train)


# In[44]:


y_pred = model.predict(x_test)


# In[45]:


y_pred


# In[46]:


y_test


# In[ ]:





# # randomforest regression

# In[77]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[78]:


df1.columns.values


# In[79]:


df1.isnull().sum()


# In[80]:


df1['Comments'] = df1['Comments'].astype(str)


# In[81]:


print(df1['Comments'].unique())
print(df1['Comments'].dtype)


# In[82]:


from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer on the comments
vectorizer.fit(df1['Comments'])

# Transform the comments into a numerical matrix
X_comments = vectorizer.transform(df1['Comments'])


# In[83]:


X_comments


# In[84]:


df1


# In[86]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df1['Title_encoded'] = label_encoder.fit_transform(df1['Title'])
df1['Video ID_encoded'] = label_encoder.fit_transform(df1['Video ID'])
df1['Published At_encoded'] = label_encoder.fit_transform(df1['Published At'])
df1['Keyword_encoded'] = label_encoder.fit_transform(df1['Keyword'])
df1['Comments_encoded'] = label_encoder.fit_transform(df1['Comments'])
df1['Views_encoded'] = label_encoder.fit_transform(df1['Views'])
df1['Comment_encoded'] = label_encoder.fit_transform(df1['Comment'])
df1['Sentiment_encoded'] = label_encoder.fit_transform(df1['Sentiment'])

X = df1[['Title_encoded', 'Video ID_encoded', 'Published At_encoded', 'Keyword_encoded', 'Comments_encoded', 'Views_encoded', 'Comment_encoded', 'Sentiment_encoded']]
y = df1['Likes1']


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[89]:


rf_model = RandomForestRegressor(random_state=42)


# In[90]:


rf_model.fit(X_train, y_train)


# In[91]:


rf_model.fit(X_train, y_train)

# Calculate the accuracy on the training set
train_accuracy = rf_model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

# Calculate the accuracy on the test set
test_accuracy = rf_model.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




