##### Import the necessary modules and models #####

import cv2
import glob
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from PIL import Image

import glob


##### Create an X list (containing images of the MRI scans) and a Y list (containing the binary labels for each MRI scan) #####

images = [cv2.imread(file) for file in glob.glob('/Users/dhsh/Desktop/archive/no/*')]

image_list = []
X = []
Y = []

for filename in glob.glob('/Users/dhsh/Desktop/archive/no/*'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
    matrix = np.asarray(im)
    matrix = np.resize(matrix, (200, 200, 3))
    Matrix1 = matrix.flatten()
    X.append(Matrix1) #X is a list of the images as matrices
    Y.append(0) #Y is the labels for the images

for filename in glob.glob('/Users/dhsh/Desktop/archive/yes/*'): #assuming gif
    im=Image.open(filename)
    image_list.append(im)
    matrix = np.asarray(im)
    matrix = np.resize(matrix, (200, 200, 3))
    Matrix1 = matrix.flatten() # Produces a one dimensional vector out of the 2d image - some models can't handle 2d images
    X.append(Matrix1)
    Y.append(1)

    
##### Apply a 70-30 split to create a training set and testing set #####

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


##### Method 1: Create 2 ML models (MLP Classifier Model and a Logistic Regression Model) ######
##### Based on their error values, modify their weights and make a prediction ######

model2 = LogisticRegression(penalty='l2', max_iter=5)
model2.fit(X_train, y_train)
logisPred = model2.predict(X_train)
logisPred2 = model2.predict(X_test)
#print(predictions)
model = MLPClassifier(max_iter=5)
model.fit(X_train, y_train)
mlpPred = model.predict(X_train)
mlpPred2 = model.predict(X_test)
#print(predictions)

mlpTrainAcc = 0
mlpTestAcc = 0
logisTrainAcc = 0
logisTestAcc = 0
for i in range(len(y_train)):
    if mlpPred[i] == y_train[i]:
        mlpTrainAcc+=1
    if logisPred[i] == y_train[i]:
        logisTrainAcc+=1

for i in range(len(y_test)):
    if mlpPred2[i] == y_test[i]:
        mlpTestAcc+=1
    if logisPred2[i] == y_test[i]:
        logisTestAcc+=1
print("MLP Training accuracy: " + str(mlpTrainAcc / len(y_train) * 100))
print("MLP Testing accuracy: " + str(mlpTestAcc / len(y_test) * 100))

print("Logistic Regression Training accuracy: " + str(logisTrainAcc / len(y_train) * 100))
print("Logistic Regression Testing accuracy: " + str(logisTestAcc / len(y_test) * 100))

acc3 = 0
for i in range(len(mlpPred2)):
    if mlpPred2[i] == logisPred2[i]:
        acc3+=1
print("Percent that both models agree on: " + str(acc3 / len(mlpPred2) * 100) + "\n\n\n\n")

w1 = 0.5
w2 = 0.5
count = len(X_test)
error1 = 0.0
error2 = 0.0
bothAcc = 0.0

for i in range(len(X_test)):
    bpred = (w1 * logisPred2[i]) + (w2 * mlpPred2[i])
    #print(str(round(bpred)) + "  " + str(bpred))
    #print("w1: " + str(w1) + "  w2: " + str(w2))
    if logisPred2[i] != y_test[i]:
        error1 += pow(0.9, count)
        w1 /= pow(2, error1)
    if mlpPred2[i] != y_test[i]:
        error2 += pow(0.9, count)
        w2 /= pow(2, error2)
    # if error1 == 1:
    #     w1 /= pow(2, error1)
    # if error2 == 1:
    #     w2 /= pow(2, error2)
    if round(bpred) == y_test[i]:
        bothAcc += 1
    sum = w1 + w2
    w1 /= sum
    w2 /= sum
    count -= 1
print((bothAcc / len(X_test)) * 100)


###### Method 2: Train a model on the training dataset. Train a second model on the data the first model predicted wrong on ######
###### Train a third model to predict which of the first two models to use depending on an MRI scan ######

linear1 = LogisticRegression(penalty='l2', max_iter=3)
linear2 = LogisticRegression(penalty='l2', max_iter=3) #Trained on the images that linear1 predicts wrong on
linearMag = LogisticRegression(penalty='l2', max_iter=3, class_weight={1:35}) #Trained to classify which model to use based on MRI Scan
linear1.fit(X_train, y_train)
l1pred = linear1.predict(X_train)
X_train2 = []
y_train2 = []
acc = accuracy_score(y_train, l1pred) * 100
print("L1 Training Accuracy: " + str(acc) + "\n")

for i in range(len(X_train)):
    if(l1pred[i] != y_train[i]): #Make a dataset with all occasions where l1 was wrong
        y_train2.append(y_train[i]) #Dataset to train linear2 on
        X_train2.append(X_train[i]) #Dataset to train linear2 on
print(len(X_train2) / len(X_train) * 100)
#print(len(y_train2) / len(X_train2))
linear2.fit(X_train2, y_train2) #Fit l2 to this dataset
l2pred = linear2.predict(X_train2)
acc2 = accuracy_score(y_train2, l2pred) * 100
print("L2 Training Accuracy: " + str(acc2) + "\n")

magLst = []
for i in range(len(X_train)): #Make a dataset for linearMag to train on
    if(l1pred[i] == y_train[i]):
        magLst.append(0) #0 means linear1
    else:
        magLst.append(1) #1 means linear2
print("MAGLST: " + str(magLst))
linearMag.fit(X_train, magLst)
magPred = linearMag.predict(X_train)
magAcc = accuracy_score(magLst, magPred) * 100
print("LMAG Training Accuracy: " + str(magAcc) + "\n")
magPredTest = linearMag.predict(X_test)

l1finalPred = []
l1finalTest = []
l2finalPred = []
l2finalTest = []
for i in range(len(X_test)):
    if(magPredTest[i] == 0):
        l1finalPred.append(X_test[i])
        l1finalTest.append(y_test[i])
    else:
        l2finalPred.append(X_test[i])
        l2finalTest.append(y_test[i])
print("MagPredTest: " + str(magPredTest))
print("l1finalPred length: " + str(len(l1finalPred)))
print("l2finalPred length: " + str(len(l2finalPred)))

l1predfinal = linear1.predict(l1finalPred)
l1finalAcc = accuracy_score(l1finalTest, l1predfinal) * 100
l2predfinal = linear2.predict(l2finalPred)
l2finalAcc = accuracy_score(l2finalTest, l2predfinal) * 100
print("Final L1 Testing Accuracy: " + str(l1finalAcc) + "\n")
print("Final L2 Testing Accuracy: " + str(l2finalAcc) + "\n")
