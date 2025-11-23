# importing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# custom neural network
import keras
from keras.models import Sequential
from keras.layers import Dense, CategoryEncoding, Dropout

# reading data set x and y into variables
xPath = "/content/drive/MyDrive/Inspirit AI Research/dataSetX.csv" # CHANGE TO YOUR SPECIFIC ADDRESS
X = pd.read_csv(xPath,index_col=0)
yPath = "/content/drive/MyDrive/Inspirit AI Research/dataSetY.csv"
y = pd.read_csv(yPath,index_col=0)

# removing MBAs to decrease runtime significantly
idx = (y["class"]!="MBA").to_numpy()
XnoMBA = X[idx]
ynoMBA = y[idx]

# DATA AUGMENTATION
# using SMOTE as data augmentation
from imblearn.over_sampling import SMOTE
# creating an instance of the smote class
sm = SMOTE(random_state=42,k_neighbors=3)
# using SMOTE on the data
X_res, y_res = sm.fit_resample(XnoMBA, ynoMBA)
# splitting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# this function will create a confusion matrix
def displayConfusionMatrix(prediction):
  # creating the matrix
  cm = confusion_matrix(y_test, prediction, labels=ynoMBA["class"].unique())
  # displaying the matrix
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ynoMBA["class"].unique())
  # plotting the matrix
  disp.plot(cmap="Reds")

# this function will train and test a model
def trainTestModel(model):
  # training
  model.fit(X_train, y_train.values.ravel())
  # testing
  prediction = model.predict(X_test)
  # accuracy
  accuracy = accuracy_score(y_test, prediction)
  # printing the accuracy
  print('Model % Accuracy: {:.2%}'.format(accuracy))
  # returning the prediction
  return prediction

# this function will run the model
def runModel(model):
  # getting the prediction
  prediction = trainTestModel(model)
  # displaying the confusion matrix
  displayConfusionMatrix(prediction)

# LOGISTIC REGRESSION
logistic_model = LogisticRegression(max_iter=500) # used to be 200 max iter
runModel(logistic_model)

# RANDOM FOREST CLASSIFIER
forest_model = RandomForestClassifier(max_depth=2, random_state=0)
runModel(forest_model)

# K-NEAREST NEIGHBOR
knn_model = KNeighborsClassifier(n_neighbors=15, algorithm='auto') # used to be 1 n neighbors
runModel(knn_model)

# NEURAL NETWORKS
mlp_model1 = MLPClassifier(hidden_layer_sizes=(10,10,10))
runModel(mlp_model1)

mlp_model1 = MLPClassifier(hidden_layer_sizes=(3,3,3))
runModel(mlp_model1)

mlp_model1 = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10))
runModel(mlp_model1)

mlp_model1 = MLPClassifier(hidden_layer_sizes=(5,5,5,5,5)) # best neural network
runModel(mlp_model1)

# setting random seed
# removes the randomness in my trials
import tensorflow as tf
import os
import random
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

# weights and biases
!pip install wandb -qU
import wandb
from wandb.keras import WandbCallback
wandb.login()

# running weights and biases
run = wandb.init(
      # Set the project where this run will be logged
      project="Asteroid Orbital Class Prediction",
      # Track hyperparameters and run metadata
      config={
      "hidden_layer_architecture" : "16,16,16,16,16",
      "num_layers" : 5,
      "num_nodes" : 16,
      "epochs": 150,
      "activation_function": "relu",
      "batch_size": 1000,
      })
config = wandb.config # use this to configure the experiement

# TRANSLATING Y VECTOR VALUES TO NUMERICAL VALUES
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_trainOneHot = ohe.fit_transform(y_train).toarray()
y_testOneHot = ohe.fit_transform(y_test).toarray()
# takes first category and sets it to index 0, takes second one and sets it to index 1, and goes on

# CREATING CUSTOM NN WITH KERAS
# n layers of x nodes
nn_model = Sequential()
nn_model.add(Dense(12, input_shape=(12,)))
for i in range(config.num_layers):
  nn_model.add(Dense(config.num_nodes, activation=config.activation_function))
nn_model.add(Dense(12, activation="softmax")) # last activation function MUST remain softmax, only used for output layer
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training and saving
modelHistory = nn_model.fit(X_train, y_trainOneHot, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_test, y_testOneHot), callbacks=[WandbCallback()])
nn_model.save("/content/drive/MyDrive/Inspirit AI Research/model.keras")

# getting training and testing accuracy
histAccuracy = modelHistory.history['accuracy']
histValAccuracy = modelHistory.history['val_accuracy']

# plotting accuracy over epochs
plt.figure()
plt.plot(histAccuracy, label="Train")
plt.plot(histValAccuracy, label="Validation")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.legend()

# finishing the run
run.finish()

# calling train function
#train()

# HYPERPARAMETER TUNING WITH SWEEPS
# creating a sweep to tune hyperparameters
sweep_config = {'method' : 'random'} # random search randomly picks valuesa and tests them

# we want to maximize the value accuracy
metric = {
    'name' : 'val_accuracy',
    'goal' : 'maximize',
}
sweep_config['metric'] = metric

# setting the values for the parameters
parameters_dict = {
    'activation_function' : {'values' : ['relu', 'tanh', 'sigmoid']},
    'num_layers' : {'values' : [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'num_nodes' : {'values' : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
    'epochs' : {'values' : [100, 125, 150, 175, 200]},
    'batch_size' : {'values' : [1000, 1250, 1500, 1750, 2000]},
}
sweep_config['parameters'] = parameters_dict

# initializing and running the sweep
sweep_id = wandb.sweep(sweep_config, project="Asteroid Orbital Class Prediction")
wandb.agent(sweep_id, train, count=100) # 6750 different parameter combinations

# this function will reverse a one hot encoded array
def undoOneHot(oneHotArr, originalValArr):
  # creating empty array for reversed values
  labels = np.empty(oneHotArr.shape[0],dtype=object)
  # for each array in the one hot encoded matrix
  for i, array in enumerate(oneHotArr):
    # get the max value index
    maxIdx = np.argmax(array)
    # turn that into the original value
    originalVal = originalValArr[maxIdx]
    # add it to the reversed value array
    labels[i] = originalVal
  # returning
  return labels

# testing
y_predOneHot = nn_model.predict(X_test)

# array for the original values before one hot encoding
orbitClassArr = ohe.categories_[0].copy()
# getting the actual predictions
y_pred = undoOneHot(y_predOneHot, orbitClassArr)

# displaying the confusion matrix
displayConfusionMatrix(y_pred)
# getting and printing accuracy
accuracy = accuracy_score(y_test, y_pred)
# printing the accuracy
print('Model % Accuracy: {:.2%}'.format(accuracy))
