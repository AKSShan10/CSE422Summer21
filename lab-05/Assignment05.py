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
#For serum_sodium column
print()
print("Handling missing values:")
# Check how many values are missing in the serum_sodium column
print("Number of rows with null values in serum_sodium column: ", heart['serum_sodium'].isnull().sum())

heart = heart[heart['serum_sodium'].notnull()]
# Print out the shape of the heart
print("Shape after removing null values in serum_sodium: ", heart.shape)

#For time column
# Check how many values are missing in the time column
print("Number of rows with null values in time column: ", heart['time'].isnull().sum())
# Subset the volunteer dataset
heart = heart[heart['time'].notnull()]

print("Shape after removing null values in time: ", heart.shape)


"""""
#Imputing missing values
#from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(heart[['serum_sodium']])

heart['serum_sodium'] = impute.transform(heart[['serum_sodium']])

#print(heart.isnull().sum())
print("After imputing values in serum_sodium column the shape is:",heart.shape)"""

#Encoding categorical features
print()
print("Encoding categorical features part:")
print(heart.info())
print(heart['sex'].unique())

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
heart['sex_enc'] = enc.fit_transform(heart['sex'])

# Compare the two columns
print(heart[['sex', 'sex_enc']].head(7))


print(heart['smoking'].unique())

enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
heart['smoking_enc'] = enc.fit_transform(heart['smoking'])

# Compare the two columns
print(heart[['smoking', 'smoking_enc']].head(7))

print(heart.head(33))
print(heart.isnull().sum())
#dropping column
heart = heart.drop(['sex'], axis = 1)
heart = heart.drop(['smoking'], axis = 1)
print("After dropping time column the shape is: ",heart.shape)

print()
print("Scaling all the values between 0-1 with proper scaling technique:")

#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

heart_scaling = heart

X_train, X_test, y_train, y_test = train_test_split(heart_scaling, heart_scaling.values, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

# transform data
X_train_scaled = scaler.transform(X_train)

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

# transform test data
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

#Feature Engineering:

print()
print("Feature Engineering:")
label = heart.DEATH_EVENT
print("label:\n{}".format(label.head(19)))
to_drop = ["DEATH_EVENT"]
features = heart.drop(to_drop, axis=1)
#print(features.head(7))


scaler = MinMaxScaler()

scaler.fit(features)

# transform data
feature_train_scaled = scaler.transform(features)
feature_train_df = pd.DataFrame(feature_train_scaled)
print(feature_train_df.head(5))