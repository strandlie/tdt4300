#!/usr/bin/env python
# coding: utf-8

# # k-Nearest Neighbors Classifier
# 
# In this notebook, you will implement your own k-nearest neighbors (k-NN) algorithm for the classification problem. You are supposed to learn:
# 
# * How to prepare the dataset for "training" and testing of the model.
# * How to implement k-nearest neighbors classification algorithm.
# * How to evaluate the performance of your classifier.
# 
# **Instructions:**
# 
# * Read carefuly through this notebook. Be sure you understand what is provided to you, and what is required from you.
# * Place your code only in sections annotated with `### START CODE HERE ###` and `### END CODE HERE ###`.
# * Use comments whenever the code is not self-explanatory.
# * Submit an executable notebook (`*.ipynb`) with your solution to BlackBoard.
# 
# Enjoy :-)
# 
# ## Packages
# 
# Following packages is all you need. Do not import any additional packages!
# 
# * [Pandas](https://pandas.pydata.org/) is a library providing easy-to-use data structures and data analysis tools.
# * [Numpy](http://www.numpy.org/) library provides support for large multi-dimensional arrays and matrices, along with functions to operate on these.

# In[4]:


import pandas as pd
import numpy as np


# ## Problem
# 
# You are given a dataset `mushrooms.csv` with characteristics/attributes of mushrooms, and your task is to implement and evaluate a k-nearest neighbors classifier able to say whether a mushroom is poisonous or edible based on its attributes.
# 
# ## Dataset
# 
# The dataset of mushroom characteristics is freely available at [Kaggle Datasets](https://www.kaggle.com/uciml/mushroom-classification) where you can find further information about the dataset. It consists of 8124 mushrooms characterized by 23 attributes (including the class). Following is the overview of attributes and values:
# 
# * class: edible=e, poisonous=p
# * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# * bruises: bruises=t,no=f
# * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# * gill-attachment: attached=a,descending=d,free=f,notched=n
# * gill-spacing: close=c,crowded=w,distant=d
# * gill-size: broad=b,narrow=n
# * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# * stalk-shape: enlarging=e,tapering=t
# * stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * veil-type: partial=p,universal=u
# * veil-color: brown=n,orange=o,white=w,yellow=y
# * ring-number: none=n,one=o,two=t
# * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# * population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# * habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# 
# Let's load the dataset into so called Pandas dataframe.

# In[5]:


mushrooms_df = pd.read_csv('mushrooms.csv')


# Now we can take a closer look at the data.

# In[6]:


mushrooms_df


# You can also print an overview of all attributes with the counts of unique values.

# In[7]:


mushrooms_df.describe().T


# The dataset is pretty much balanced. That's a good news for the evaluation.

# ## Dataset Preprocessing
# 
# As our dataset consist of nominal/categorical values only, we will encode the strings into integers which will allow us to use similiraty measures such as Euclidean distance.

# In[8]:


def encode_labels(df):
    import sklearn.preprocessing
    encoder = {}
    for col in df.columns:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoder[col] = le
    return df, encoder    

mushrooms_encoded_df, encoder = encode_labels(mushrooms_df)


# In[9]:


mushrooms_encoded_df


# ## Dataset Splitting
# 
# Before we start with the implementation of our k-nearest neighbors algorithm we need to prepare our dataset for the "training" and testing.
# 
# First, we divide the dataset into attributes (often called features) and classes (often called targets). Keeping attributes and classes separately is a common practice in many implementations. This should simplify the implementation and make the code understandable.

# In[10]:


X_df = mushrooms_encoded_df.drop('class', axis=1)  # attributes
y_df = mushrooms_encoded_df['class']  # classes
X_array = X_df.as_matrix()
y_array = y_df.as_matrix()


# And this is how it looks like.

# In[11]:


print('X =', X_array)
print('y =', y_array)


# Next, we need to split the attributes and classes into training sets and test sets.
# 
# **Exercise:**
# 
# Implement the holdout splitting method with shuffling.

# In[12]:


def train_test_split(X, y, test_size=0.2):
    """
    Shuffles the dataset and splits it into training and test sets.
    
    :param X
        attributes
    :param y
        classes
    :param test_size
        float between 0.0 and 1.0 representing the proportion of the dataset to include in the test split
    :return
        train-test splits (X-train, X-test, y-train, y-test)
    """
    ### START CODE HERE ###
    X_train = np.empty(shape=(0, len(X[0])))
    X_test = np.empty(shape=(0,len(X[0])))
    y_train = np.empty(0)
    y_test = np.empty(0)

    while len(y) > 0:
        sample_index = np.random.randint(0, len(y))

        input_X = X[sample_index].reshape(1, -1)
        X_train = np.concatenate((X_train, input_X))
        y_train = np.append(y_train, y[sample_index])
        
        X = np.delete(X, sample_index, 0)
        y = np.delete(y, sample_index)

    
    num_of_test = int(round(len(y_train) * test_size))
    
    X_test = X_train[:num_of_test]
    X_train = X_train[num_of_test:]
    y_test = y_train[:num_of_test]
    y_train = y_train[num_of_test:]
    ### END CODE HERE ###
    return X_train, X_test, y_train, y_test


# Let's split the dataset into training and validation/test set with 67:33 split.

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)


# In[14]:


print('X_train =', X_train)
print('y_train =', y_train)
print('X_test =', X_test)
print('y_test =', y_test)


# A quick sanity check...

# In[15]:


assert len(X_train) == len(y_train)
assert len(y_train) == 5443
assert len(X_test) == len(y_test)
assert len(y_test) == 2681


# ## Algorithm
# 
# The k-nearest neighbors algorithm doesn't require a training step. The class of an unseen sample is deduced by comparison with samples of known class.
# 
# **Exercise:**
# 
# Implement the k-nearest neighbors algorithm.

# In[ ]:


# Use this section to place any "helper" code for the `knn()` function.

### START CODE HERE ###

### END CODE HERE ###


# In[18]:


def knn(X_true, y_true, X_pred, k=5):
    """
    k-nearest neighbors classifier.
    
    :param X_true
        attributes of the groung truth (training set)
    :param y_true
        classes of the groung truth (training set)
    :param X_pred
        attributes of samples to be classified
    :param k
        number of neighbors to use
    :return
        predicted classes
    """
    ### START CODE HERE ### 
    y_pred = np.empty(0)
    
    #  Iterate over all x in X_pred, find the distances and save them in 
    #  the dictionary distances. Sort the keys in distances and extract.
    #np.array(shape=(len(X_pred), len(X_true)))
    
    for x in X_pred:
        distances = {}
        for i in range(len(X_true)):
            dist = np.linalg.norm(X_true[i] - x)
            distances[dist] = distances.get(dist, []) + [i]
        
        shortest_distances = list(distances.keys())
        shortest_distances.sort()
        
        j = 0
        X_pred_indices = []
        while len(X_pred_indices) < k:
            X_pred_indices = X_pred_indices + distances[shortest_distances[j]]
            j += 1
            
        X_pred_indices = X_pred_indices[:k]
        
        num_of_1 = 0
        for index in X_pred_indices:
            if y_true[index] == 1:
                num_of_1 += 1
                
        if num_of_1 / len(X_pred_indices) >= 0.5:
            y_pred = np.append(y_pred, 1.0)
        else:
            y_pred = np.append(y_pred, 0.0)
            
            

    ### END CODE HERE ### 
    return y_pred


# In[19]:


y_hat = knn(X_train, y_train, X_test, k=5)


# First ten predictions of the test set.

# In[ ]:


y_hat[:10]


# ## Evaluation
# 
# Now we would like to assess how well our classifier performs.
# 
# **Exercise:**
# 
# Implement a function for calculating the accuracy of your predictions given the ground truth and predictions.

# In[ ]:


def evaluate(y_true, y_pred):
    """
    Function calculating the accuracy of the model on the given data.
    
    :param y_true
        true classes
    :paaram y
        predicted classes
    :return
        accuracy
    """
    ### START CODE HERE ### 
    correct = 0.0
    total = 0.0
    for i in range(len(y_pred)):
        if np.isclose(y_pred[i], y_true[i]):
            correct += 1
        total += 1
    
    accuracy = correct / total
    ### END CODE HERE ### 
    return accuracy


# In[ ]:


accuracy = evaluate(y_test, y_hat)
print('accuracy =', accuracy)


# How many items where misclassified?

# In[ ]:


print('misclassified =', sum(abs(y_hat - y_test)))


# How balanced is our test set?

# In[ ]:


np.bincount(y_test.astype('int64'))


# If it's balanced, we don't have to be worried about objectivity of the accuracy metric.

# ---
# 
# Congratulations! At this point, hopefully, you have successufuly implemented a k-nearest neighbors algorithm able to classify unseen samples with high accuracy.
# 
# ✌️
