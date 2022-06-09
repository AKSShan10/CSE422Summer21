'''
1. Collect the dataset that you have used to complete the Lab 5 assignment, load it using pandas

2. Apply necessary pre-processing steps on it

3. Support Vector Machine (SVM), Neural Network (Multilayer Perceptron Classifier) and Random Forest are three
very popular machine learning classifiers. Divide the dataset into 8:2 train-test split and perform
Support Vector Machine, Neural Network (MLPClassifier) and Random Forest on it using sklearn library.
In the previous assignment, you have already used Logistic Regression and decision tree classifiers from
the sklearn library. Just change the imports and the function calls to use other classifiers.
Take a look at the sklearn documentation for further information.

4. Perform dimensionality reduction using PCA. Reduce the number of feature vectors into half
(e.g. if your dataset has 10 columns, after applying PCA it should have 5 columns)

5.Apply Support Vector Machine, Neural Network (MLPClassifier) and Random Forest again on the reduced dataset.

6. Compare the accuracy of the pre-PCA and post-PCA results. Plot them against each other in a bar graph.

7. Copy all your code in a doc file and submit the doc file
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib
inline

glass = pd.read_csv("/content/glass source classification dataset.csv")
glass = glass.drop(axis=1, columns='Unnamed: 0')
glass.head(3)
glass.shape
glass.isnull().sum()
'''
no need to remove any col
volunteer = volunteer.drop(['BIN', 'BBL', 'NTA'], axis = 1)
volunteer.shape
'''
# Check how many values are missing in the category_desc column
print("Number of rows with null values in Ca column: ", glass['Ca'].isnull().sum())

# Subset the volunteer dataset

glass_subset = glass[glass['Ca'].notnull()]

# Print out the shape of the subset
'''
no need to remove null value
print("Shape after removing null values: ", volunteer_subset.shape)
print("Shape of dataframe before dropping:", volunteer.shape)
volunteer = volunteer.dropna(axis = 0, subset = ['Ca'])
print("Shape after dropping:", volunteer.shape)
'''
# volunteer.fillna(50)// no going to set directly number
from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(glass[['Ca']])

glass['Ca'] = impute.transform(glass[['Ca']])
glass
glass.info()
glass["Ba"].unique()
glass["Fe"].unique()
glass["Type"].unique()
from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
glass['Ba'] = enc.fit_transform(glass['Ba'])
glass['Fe'] = enc.fit_transform(glass['Fe'])
glass['Type'] = enc.fit_transform(glass['Type'])
features = glass.drop(axis=1, columns='Type')
features.head(40)

label = glass[['Type']]
label.head(80)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(features)
feature_scaled = scaler.transform(features)
feature_scaled_df = pd.DataFrame(feature_scaled, columns=list(glass.columns)[:-1])
feature_scaled_df.head(50)
features = glass.drop(axis=1, columns='Type')
glass.head(40)

import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Prepare the training set
data = glass
# X = feature values, all the columns except the last column
x = data.iloc[:, :-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]
# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=42)
# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)  # Training the model
predictions = model.predict(x_test)
print(predictions)  # printing predictions

# l=[]
# from sklearn.svm import SVC
# svc = SVC(kernel="linear")
# svc.fit(x_train, y_train)
# print("Training accuracy of the model is {:.2f}".format(svc.score(x_train, y_train)))
# print("Testing accuracy of the model is {:.2f}".format(svc.score(x_test, y_test)))
# predictions1 = svc.predict(x_test)
# print(predictions1)
# l.append(predictions1)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC(kernel="linear")
svc.fit(x_train, y_train)
predictions_svc_pre = svc.predict(x_test)
accuracy_svc_pre = accuracy_score(y_test, predictions_svc_pre)
accuracy_svc_pre

# from sklearn.neural_network import MLPClassifier
# #classifier = MLPClassifier(kernel='rbf', random_state = 1)
# nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=1000000)
# nnc.fit(x_train, y_train)
# print("The Training accuracy of the model is {:.2f}".format(nnc.score(x_train, y_train)))
# print("The Testing accuracy of the model is {:.2f}".format(nnc.score(x_test, y_test)))
# predictions2 = nnc.predict(x_test)
# print(predictions2)
# l.append(predictions2)
# from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)
predictions_random_tree_pre = rfc.predict(x_test)
accuracy_random_forest_pre = accuracy_score(y_test, predictions_random_tree_pre)
accuracy_random_forest_pre

# Create a Gaussian Classifier
# rfc = RandomForestClassifier(n_estimators=50)
# rfc.fit(x_train, y_train)
# print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
# print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))

# predictions3 = rfc.predict(x_test)
# print(predictions3)
# l.append(predictions3)
from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(hidden_layer_sizes=(9), activation="relu", max_iter=10000)
nnc.fit(x_train, y_train)
predictions_neural_pre = nnc.predict(x_test)
accuracy_multilayer_pre = accuracy_score(y_test, predictions_neural_pre)
accuracy_multilayer_pre


from sklearn.decomposition import PCA

feature = glass
length_of_feature_column = feature.shape[1]
pca = PCA(n_components=int(length_of_feature_column / 2))
principal_components = pca.fit_transform(feature.values)
principal_df = pd.DataFrame(data=principal_components,
                            columns=["principle component 1", "principle component 2", "principle component 3",
                                     "principle component 4", "principle component 5"])
main_df = pd.concat([principal_df, label], axis=1)
main_df

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(principal_df, label, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC(kernel="linear")
svc.fit(x_train, y_train)
predictions_svc_post = svc.predict(x_test)
accuracy_svc_post = accuracy_score(y_test, predictions_svc_post)
accuracy_svc_post

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)
predictions_random_tree_post = rfc.predict(x_test)
accuracy_random_forest_post = accuracy_score(y_test, predictions_random_tree_post)
accuracy_random_forest_post

from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(hidden_layer_sizes=(9), activation="relu", max_iter=10000)
nnc.fit(x_train, y_train)
predictions_neural_post = nnc.predict(x_test)
accuracy_multilayer_post = accuracy_score(y_test, predictions_neural_post)
accuracy_multilayer_post

accuracy_df = pd.DataFrame(
    {"accuracy_name": ["Ac.svc.pre", "Ac.svc.post", "Ac.ran.pre", "Ac.ran.post", "Ac.mult.pre", "Ac.mult.post"],
     "category": [accuracy_svc_pre, accuracy_svc_post, accuracy_random_forest_pre, accuracy_random_forest_post,
                  accuracy_multilayer_pre, accuracy_multilayer_post],
     "Type": ["svc", "svc", "random", "random", "neural", "neural"]})
accuracy_df

import seaborn as sns
from matplotlib import pyplot as plt

plt.title("accuracy_comparision")
sns.set(rc={'figure.figsize': (20, 8.27)})
sns.barplot(x="accuracy_name", y="category", data=accuracy_df, hue="Type", palette="Blues_d", dodge=False)
'''
pca = PCA(n_components=int(feature.shape[1]/2))
principal_components= pca.fit_transform(feature.values)
principal_df = pd.DataFrame(data=principal_components, columns=["principle component 1", "principle component 2", "principle component 3", "principle component 4", "principle component 5"])
df_result=pd.concat([principal_df, label], axis=1)
df_result


import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#Prepare the training set
data =df_result
# X = feature values, all the columns except the last column
x1 = data.iloc[:, :-1]

# y = target values, last column of the data frame
y1 = data.iloc[:, -1]
#Split the data into 80% training and 20% testing
x1_train, x1_test, y1_train, y1_test = train_test_split(x1.values, y1.values, test_size=0.2, random_state=42)
accuracy_df=[]
#Train the model
# model = LogisticRegression()
# model.fit(x_train, y_train) #Training the model
# predictions = model.predict(x_test)
# print(predictions)# printing predictions



from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x1_train, y1_train)
print("Training accuracy of the model is {:.2f}".format(svc.score(x1_train, y1_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(x1_test, y1_test)))
predictions10 = svc.predict(x1_test)
print(predictions10)
l.append(predictions10)
from sklearn.neural_network import MLPClassifier
#classifier = MLPClassifier(kernel='rbf', random_state = 1)
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=1000000)
nnc.fit(x1_train, y1_train)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(x1_train, y1_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(x1_test, y1_test)))
predictions20 = nnc.predict(x1_test)
print(predictions20)
l.append(predictions20)
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x1_train, y1_train)
print("The Training accuracy of the model is {:.2f}".format(rfc.score(x1_train, y1_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x1_test, y1_test)))

predictions30 = rfc.predict(x1_test)
print(predictions30)
l.append(predictions30)
accuracy_df = pd.DataFrame({'accuracy_name':["Accuracy svc pre","Accuracy svc post", "Accuracy multilayer pre", "Accuracy_multilayer post", "Accuracy_random_tree pre", 
                        "Accuracy_random_tree post"], 
             "category": [predictions1, predictions10, predictions3, predictions30, predictions2, predictions20] , 
              "Type":["svc", "svc", "random", "random", "neural", "neural"]})
accuracy_df
accuracy_df["category"]=accuracy_df["category"].astype('float')

import seaborn as sns
from matplotlib import pyplot as plt
plt.title("Accuracy Comparision")
sns.set(rc={'figure.figsize':(16,8.27)})
sns.barplot(x="accuracy_name", y = "category", data = accuracy_df, hue = "Type")
'''
