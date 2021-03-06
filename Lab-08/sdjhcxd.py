# -*- coding: utf-8 -*-
"""Welcome to Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb

<p><img alt="Colaboratory logo" height="45px" src="/img/colab_favicon.ico" align="left" hspace="10px" vspace="0px"></p>

<h1>What is Colaboratory?</h1>

Colaboratory, or 'Colab' for short, allows you to write and execute Python in your browser, with
- Zero configuration required
- Free access to GPUs
- Easy sharing

Whether you're a <strong>student</strong>, a <strong>data scientist</strong> or an <strong>AI researcher</strong>, Colab can make your work easier. Watch <a href="https://www.youtube.com/watch?v=inN8seMm7UI">Introduction to Colab</a> to find out more, or just get started below!

## <strong>Getting started</strong>

The document that you are reading is not a static web page, but an interactive environment called a <strong>Colab notebook</strong> that lets you write and execute code.

For example, here is a <strong>code cell</strong> with a short Python script that computes a value, stores it in a variable and prints the result:
"""

seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day

"""To execute the code in the above cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut 'Command/Ctrl+Enter'. To edit the code, just click the cell and start editing.

Variables that you define in one cell can later be used in other cells:
"""

seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week

"""Colab notebooks allow you to combine <strong>executable code</strong> and <strong>rich text</strong> in a single document, along with <strong>images</strong>, <strong>HTML</strong>, <strong>LaTeX</strong> and more. When you create your own Colab notebooks, they are stored in your Google Drive account. You can easily share your Colab notebooks with co-workers or friends, allowing them to comment on your notebooks or even edit them. To find out more, see <a href="/notebooks/basic_features_overview.ipynb">Overview of Colab</a>. To create a new Colab notebook you can use the File menu above, or use the following link: <a href="http://colab.research.google.com#create=true">Create a new Colab notebook</a>.

Colab notebooks are Jupyter notebooks that are hosted by Colab. To find out more about the Jupyter project, see <a href="https://www.jupyter.org">jupyter.org</a>.

## Data science

With Colab you can harness the full power of popular Python libraries to analyse and visualise data. The code cell below uses <strong>numpy</strong> to generate some random data, and uses <strong>matplotlib</strong> to visualise it. To edit the code, just click the cell and start editing.
"""
"""""
import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()"""

"""You can import your own data into Colab notebooks from your Google Drive account, including from spreadsheets, as well as from GitHub and many other sources. To find out more about importing data, and how Colab can be used for data science, see the links below under <a href="#working-with-data">Working with data</a>.

## Machine learning

With Colab you can import an image dataset, train an image classifier on it, and evaluate the model, all in just <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb">a few lines of code</a>. Colab notebooks execute code on Google's cloud servers, meaning you can leverage the power of Google hardware, including <a href="#using-accelerated-hardware">GPUs and TPUs</a>, regardless of the power of your machine. All you need is a browser.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

desired_width = 320
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns',14)


heart = pd.read_csv("E:\SEMESTER\Summer- 21\CSE422     ARTIFICIAL INTELLIGENCE\Lab\Lab-08\heart failur classification dataset.csv")
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

# y = target values,  last column of the data frame
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
print(accuracy_score_of_pre_nnc)
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
accuracy_score_of_pre_rfc = accuracy_score(rfc_predictions_pre,y_test)
print(accuracy_score_of_pre_rfc)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))

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

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print("svcpost")

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(x_train, y_train.ravel())
svc_predictions_post = svc.predict(x_test)
print("prediction:",svc_predictions_post)# printing predictions

accuracy_score_of_post_svc = accuracy_score(svc_predictions_post,y_test)
print("svc accuracy:", accuracy_score_of_post_svc)

print("Training accuracy of the model is {:.2f}".format(svc.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(svc.score(x_test, y_test)))



print("### Neural Network Classifier/MLPClassifier")
from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(x_train, y_train)
nnc_predictions_post = nnc.predict(x_test)
print("predictions:",nnc_predictions_pre)
accuracy_score_of_post_nnc = accuracy_score(nnc_predictions_post,y_test)
print("The Training accuracy of the model is {:.2f}".format(nnc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(nnc.score(x_test, y_test)))
#predictions = nnc.predict(x_test)
#print(predictions)

print("RandomForestClassifier:")
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train.ravel())
rfc_predictions_post = rfc.predict(x_test)

print("predictions:",rfc_predictions_post)
accuracy_score_of_post_rfc = accuracy_score(rfc_predictions_post,y_test)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(x_train, y_train)))
print("The Testing accuracy of the model is {:.2f}".format(rfc.score(x_test, y_test)))



import matplotlib.pyplot as plt
#colors = {'logistic regression':'r', 'decision tree':'g'}
fig,ax = plt.subplots()
ax.bar(['accuracy score of pre-pca svc','accuracy score of post-pca svc'],[accuracy_score_of_pre_svc,accuracy_score_of_post_svc])
ax.set_title('Comparison of pre-PCA and post-PCA result of SVC part')
ax.set_xlabel("Method name")
ax.set_ylabel('Accuracy')
plt.show()

import matplotlib.pyplot as plt
#colors = {'logistic regression':'r', 'decision tree':'g'}
fig,ax = plt.subplots()
ax.bar(['accuracy score of pre-pca nnc','accuracy score of  post-pca nnc'],[accuracy_score_of_pre_nnc,accuracy_score_of_post_nnc])
ax.set_title('Comparison of pre-PCA and post-PCA result of MLPClassifier part')
ax.set_xlabel("Method name")
ax.set_ylabel('Accuracy')
plt.show()



import matplotlib.pyplot as plt
#colors = {'logistic regression':'r', 'decision tree':'g'}
fig,ax = plt.subplots()
ax.bar(['accuracy score of pre-pca rfc','accuracy score of post-pca rfc'],[accuracy_score_of_pre_rfc,accuracy_score_of_post_rfc])
ax.set_title('Comparison of pre-PCA and post-PCA result of RandomForestClassifier part')
ax.set_xlabel("Method name")
ax.set_ylabel('Accuracy')
plt.show()

"""Colab is used extensively in the machine learning community with applications including:
- Getting started with TensorFlow
- Developing and training neural networks
- Experimenting with TPUs
- Disseminating AI research
- Creating tutorials

To see sample Colab notebooks that demonstrate machine learning applications, see the <a href="#machine-learning-examples">machine learning examples</a> below.

## More resources

### Working with notebooks in Colab
- [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb)
- [Guide to markdown](/notebooks/markdown_guide.ipynb)
- [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
- [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
- [Interactive forms](/notebooks/forms.ipynb)
- [Interactive widgets](/notebooks/widgets.ipynb)
- <img src="/img/new.png" height="20px" align="left" hspace="4px" alt="New"></img>
 [TensorFlow 2 in Colab](/notebooks/tensorflow_version.ipynb)

<a name="working-with-data"></a>
### Working with data
- [Loading data: Drive, Sheets and Google Cloud Storage](/notebooks/io.ipynb) 
- [Charts: visualising data](/notebooks/charts.ipynb)
- [Getting started with BigQuery](/notebooks/bigquery.ipynb)

### Machine learning crash course
These are a few of the notebooks from Google's online machine learning course. See the <a href="https://developers.google.com/machine-learning/crash-course/">full course website</a> for more.
- [Intro to Pandas](/notebooks/mlcc/intro_to_pandas.ipynb)
- [TensorFlow concepts](/notebooks/mlcc/tensorflow_programming_concepts.ipynb)

<a name="using-accelerated-hardware"></a>
### Using accelerated hardware
- [TensorFlow with GPUs](/notebooks/gpu.ipynb)
- [TensorFlow with TPUs](/notebooks/tpu.ipynb)

<a name="machine-learning-examples"></a>

## Machine learning examples

To see end-to-end examples of the interactive machine-learning analyses that Colaboratory makes possible, take a look at these tutorials using models from <a href="https://tfhub.dev">TensorFlow Hub</a>.

A few featured examples:

- <a href="https://tensorflow.org/hub/tutorials/tf2_image_retraining">Retraining an Image Classifier</a>: Build a Keras model on top of a pre-trained image classifier to distinguish flowers.
- <a href="https://tensorflow.org/hub/tutorials/tf2_text_classification">Text Classification</a>: Classify IMDB film reviews as either <em>positive</em> or <em>negative</em>.
- <a href="https://tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization">Style Transfer</a>: Use deep learning to transfer style between images.
- <a href="https://tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa">Multilingual Universal Sentence Encoder Q&amp;A</a>: Use a machine-learning model to answer questions from the SQuAD dataset.
- <a href="https://tensorflow.org/hub/tutorials/tweening_conv3d">Video Interpolation</a>: Predict what happened in a video between the first and the last frame.
"""