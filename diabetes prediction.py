import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading a diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv("diabetes.csv")

# printing the first file 5 rows of the dataset
diabetes_dataset.head()

# number of rows and columns in this dataset
diabetes_dataset.shape

# getting the statistical measure of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# separating data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']



scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)



X = standardized_data
Y = diabetes_dataset['Outcome']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)



classifier = svm.SVC(kernel='linear')

# training the support vector machine classifier
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('accuracy score of the traing data: ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('accuracy score of the test data: ', test_data_accuracy)

input_data = name = input("Enter details of patient :Pregnancies, Glucose,BloodPressure, SkinThickness, Insulin, BMI, "
                          "DiabetesPedigreeFunction, Age, Outcome")

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are prediction for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input-data
std_data = scaler.transform(input_data_reshaped)


prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('the person is not diabetic')
else:
    print('the person is diabetic')
