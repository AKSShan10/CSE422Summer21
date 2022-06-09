import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

desired_width = 320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',14)


heart = pd.read_csv('E:\SEMESTER\Summer- 21\CSE422     ARTIFICIAL INTELLIGENCE\Lab\Lab-08\heart failur classification dataset.csv')
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
heart['sex'] = enc.fit_transform(heart['sex'])

# Compare the two columns
#print(heart[['sex', 'sex_enc']].head(7))

#print(heart['smoking'].unique())

enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
heart['smoking'] = enc.fit_transform(heart['smoking'])

# Compare the two columns
#print(heart[['smoking', 'smoking_enc']].head(7))

print(heart.head(33))
print(heart.isnull().sum())
#dropping column

#print("Scaling all the values between 0-1 with proper scaling technique:")

#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

label = heart[['DEATH_EVENT']]
features = heart.drop(axis=1, columns='DEATH_EVENT')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(features)
feature_scaled = scaler.transform(features)
feature_scaled_df = pd.DataFrame(feature_scaled, columns=list(heart.columns)[:-1])
print(feature_scaled_df.head())
print(label.head())


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#heart = heart_scaling
#print(heart.head())
#Prepare the training set
# Perform classification and calculate accuracy using logistic regression
#heart = dataset

# X = feature values, all the columns except the   last column
X = heart.iloc[:,:-1].values

# y = target values, 3rd last column of the data frame
y = heart.iloc[:,-1].values
#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
predictions = model.predict(x_test)
print("prediction:",predictions)# printing predictions

accuracy_score_of_regression = accuracy_score( predictions,y_test)
print("logistic regression accuracy:", accuracy_score_of_regression)

#Perform classification and calculate accuracy using decision tree

from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import train_test_split
X = heart.iloc[:, :-1].values
y = heart.iloc[:,-1].values
print(X.shape)
print("y",y.shape)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
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


print("svc:")

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x_train, y_train)
svc_predictions_pre = svc.predict(x_test)
print("prediction:",svc_predictions_pre)# printing predictions

accuracy_score_of_pre_svc = accuracy_score(svc_predictions_pre,y_test)
print("svc accuracy:", accuracy_score_of_pre_svc)

print("Training accuracy of the model is {:.2f}".format(svc.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(x_test, y_test)))

print("### Neural Network Classifier/MLPClassifier")
from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(x_train, y_train)
nnc_predictions_pre = nnc.predict(x_test)
print("predictions:",nnc_predictions_pre)
accuracy_score_of_pre_nnc = accuracy_score(nnc_predictions_pre,y_test)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(x_test, y_test)))
#predictions = nnc.predict(x_test)
#print(predictions)

print("RandomForestClassifier")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)
rfc_predictions_pre = rfc.predict(x_test)

print("predictions:",rfc_predictions_pre)
accuracy_score_of_pre_nnc = accuracy_score(rfc_predictions_pre,y_test)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))

#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)
#data=pca.fit_transform(scaler.fit_transform(mnist.data))

print("pca")
from sklearn.decomposition import PCA
pca = PCA(n_components = features.shape[1] // 2)

principal_components= pca.fit_transform(features.values)
print(principal_components)

print("sum",sum(pca.explained_variance_ratio_))

principal_df = pd.DataFrame(data=principal_components, columns=["component 1", "component 2","component 3","component 4","component 5","component 6"])


print(principal_df.head())

main_df = pd.concat([principal_df, label], axis=1)
print("main_df:",main_df.head())

print("After implementing PCA")
X=principal_df.values
y=label.values
"""""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(principal_df,label, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC(kernel="linear")
svc.fit(x_train, y_train)
predictions_svc_post = svc.predict(x_test)
accuracy_svc_post = accuracy_score(y_test, predictions_svc_post)
accuracy_svc_post"""
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


print("svcpost")

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x_train, y_train.ravel())
svc_predictions_post = svc.predict(x_test)
print("prediction:",svc_predictions_post)# printing predictions

accuracy_score_of_post_svc = accuracy_score(svc_predictions_post,y_test)
print("svc accuracy:", accuracy_score_of_pre_svc)

print("Training accuracy of the model is {:.2f}".format(svc.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(x_test, y_test)))

print("RandomForestClassifier:")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train.ravel())
rfc_predictions_post = rfc.predict(x_test)

print("predictions:",rfc_predictions_post)
accuracy_score_of_pre_nnc = accuracy_score(rfc_predictions_post,y_test)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))










