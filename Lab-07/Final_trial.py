import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
"""""
desired_width = 320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',14)
"""

heart = pd.read_csv('E:\SEMESTER\Summer- 21\CSE422     ARTIFICIAL INTELLIGENCE\Lab\lab-05\heart failur classification dataset.csv')
#heart.insull()
#printing 1st 7 rows
print(heart.head(7))
#printing total rows & column
print(heart.shape)
# To get the column info with missing data
print(heart.isnull().sum())

#Handling missing values
#print("Handling missing values:")
# Check how many values are missing in the serum_sodium column
print("Number of rows with null values in serum_sodium column: ", heart['serum_sodium'].isnull().sum())

heart = heart[heart['serum_sodium'].notnull()]
# Print out the shape of the heart
print("Shape after removing null values in serum_sodium: ", heart.shape)

# Check how many values are missing in the time column
#print("Number of rows with null values in time column: ", heart['time'].isnull().sum())
heart = heart[heart['time'].notnull()]

print("Shape after removing null values in time: ", heart.shape)

#Encoding categorical features
#print("Encoding categorical features part:")
#print(heart.info())
#print(heart['sex'].unique())

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
heart['sex_enc'] = enc.fit_transform(heart['sex'])

# Compare the two columns
#print(heart[['sex', 'sex_enc']].head(7))

#print(heart['smoking'].unique())

enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
heart['smoking_enc'] = enc.fit_transform(heart['smoking'])

# Compare the two columns
#print(heart[['smoking', 'smoking_enc']].head(7))

print(heart.head(33))
print(heart.isnull().sum())
#dropping column
heart = heart.drop(['sex'], axis = 1)
heart = heart.drop(['smoking'], axis = 1)
#print("After dropping time column the shape is: ",heart.shape)

#print("Scaling all the values between 0-1 with proper scaling technique:")

#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

heart_scaling = heart

X_train, X_test, y_train, y_test = train_test_split(heart_scaling, heart_scaling.values, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

# transform data
X_train_scaled = scaler.transform(X_train)

# transform test data
X_test_scaled = scaler.transform(X_test)
#print(X_test_scaled)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Prepare the training set
# Perform classification and calculate accuracy using logistic regression
#heart = dataset

# X = feature values, all the columns except the 3rd last column
X = heart.iloc[:, :-3].values

# y = target values, 3rd last column of the data frame
y = heart.iloc[:, -3].values
#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
predictions = model.predict(x_test)
print("prediction:",predictions)# printing predictions

accuracy_score_of_regression = accuracy_score(y_test, predictions)
print("logistic regression accuracy:", accuracy_score_of_regression)

#Perform classification and calculate accuracy using decision tree

from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
X = heart.iloc[:, :-3].values
y = heart.iloc[:,-3].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=accuracy_score(y_pred,y_test)
print("Decission tree accuracy:",score)

#graphical representation
import matplotlib.pyplot as plt
#colors = {'logistic regression':'r', 'decision tree':'g'}
fig,ax = plt.subplots()
ax.bar(['logistic regression','decision tree'],[accuracy_score_of_regression,score])
ax.set_title('Comparison of logistic regression and decision tree')
ax.set_xlabel("Method name")
ax.set_ylabel('Accuracy')
plt.show()
