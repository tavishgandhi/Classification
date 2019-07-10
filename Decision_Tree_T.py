# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:54:45 2019

@author: Tavish Gandhi
"""
'''
The Dataset is about predicting which passengers had higher chance of survival
in Titanic tragedy,we have our target column Survived in Y and independent variables 
in X.

We will use decision Tree for this classifiction problem.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Titanic.csv")

#Removing unwanted columns
df = dataset.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis = 'columns')
y = pd.DataFrame(df["Survived"])
X = df.drop(['Survived'],axis = 'columns')
X.head()

# Label Encode the text columns
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
X["Sex_N"] = le_sex.fit_transform(X["Sex"])

# Since we already Label Encoded Sex column,we can drop the Text Sex Column
X = X.drop(["Sex"],axis = 'columns')

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
Age_n = X.iloc[:,1:2].values # Making a new Dataframe and adjusting missing values there
imputer = imputer.fit(Age_n)
Age_n = imputer.transform(Age_n)
X["Age_n"] = Age_n # Making a new column with name Age_n and copying values from Age_n Dataframe
X = X.drop(["Age"],axis = "columns") # Droping the original Age column with missing values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Making our Decision Tree model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
y_pred = model.predict(X_test)
print("Accuracy on training set {:.3f}".format(model.score(X_train,y_train)))
print("Accuracy on test set {:.3f}".format(model.score(X_test,y_test)))

#Improving model by limiting depth
model_n = tree.DecisionTreeClassifier(max_depth = 5, random_state = 0)
model_n.fit(X_train,y_train)
print("Accuracy on training set {:.3f}".format(model_n.score(X_train,y_train)))
print("Accuracy on test set {:.3f}".format(model_n.score(X_test,y_test)))


# Visualising the Decision Tree results 
import graphviz
from sklearn.tree import export_graphviz
export_graphviz(decision_tree=model_n, out_file = "DT.dot" , feature_names = ['Pclass','Fare','Sex_N','Age_n'], class_names = ['Survived','Dead'],impurity = False,filled = True)
# To convert the dot file to png at cmd,goto working directory then use "dot -Tpng  DT.dot -o DT.png"


# We have methods in Decision Tree which allow us to look at important features..
print("Feature Importances {}".format(model_n.feature_importances_))
''' Above here gives us an array of importance of features in range 0 to 1'''


# We will visualize the importance of features with help of bar graph...
'''One thing to note is that bar graph doesnot entertain text so we will convert into int
    One way to visualize as per tutorial
'''

n_features = len(X_train.columns)# Number of features
print(n_features)
feature_names = X_train.columns
plt.barh(range(n_features), model_n.feature_importances_,align = 'center')
plt.yticks(np.arange(n_features),feature_names)# Since we earlier only used int,yticks are used to convert int into labels
plt.xlabel("Feature importance")
plt.ylabel("Features")
plt.show()

''' Visualizing A bar graph'''

xpos = np.arange(len(X_train.columns))
plt.bar(xpos,model_n.feature_importances_)
# When you look at bar graph ,X axis contains features but in integer format,we will convert that back using xticks
plt.xticks(xpos,X_train.columns)# The 2 arguments,first array of int and 2nd corresponding feature 
plt.xlabel("Features")
plt.ylabel("Feature importances")
plt.show()


