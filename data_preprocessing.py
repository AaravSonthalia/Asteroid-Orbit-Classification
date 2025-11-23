# importing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# reading data set into variable
dataSetPath = "/content/drive/MyDrive/Inspirit AI Research/dataset.csv" # CHANGE TO YOUR SPECIFIC ADDRESS
dataSet = pd.read_csv(dataSetPath)
processedDataSet = dataSet.copy()

# removing unecessary features from data set
processedDataSet.drop(['id','spkid','full_name','pdes','name','prefix','neo','pha','diameter_sigma','orbit_id','epoch'], axis=1, inplace=True)
processedDataSet.drop(['equinox','tp','tp_cal','per_y','om','moid','moid_ld','sigma_e','sigma_a','sigma_q','sigma_i','sigma_om'], axis=1, inplace=True)
processedDataSet.drop(['sigma_ad','sigma_n','sigma_tp','sigma_per','rms','sigma_w','sigma_ma','epoch_mjd','epoch_cal'], axis=1, inplace=True)

# changing null values to median values
np.sum(processedDataSet.isnull())
processedDataSet['H'].fillna(processedDataSet['H'].median(), inplace=True)
processedDataSet['diameter'].fillna(processedDataSet['diameter'].median(), inplace=True)
processedDataSet['albedo'].fillna(processedDataSet['albedo'].median(), inplace=True)
processedDataSet['ma'].fillna(processedDataSet['ma'].median(), inplace=True)
processedDataSet['ad'].fillna(processedDataSet['ad'].median(), inplace=True)
processedDataSet['per'].fillna(processedDataSet['per'].median(), inplace=True)

# PREPARING DATA FOR MODEL
# setting y vector and x matrix
y = processedDataSet['class']
x = processedDataSet.drop(columns='class')

# reading the x and y to a file
x.to_csv("/content/drive/MyDrive/Inspirit AI Research/dataSetX.csv") # CHANGE TO YOUR SPECIFIC ADDRESS
y.to_csv("/content/drive/MyDrive/Inspirit AI Research/dataSetY.csv")
