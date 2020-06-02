from sklearn.svm import SVC, LinearSVC
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from skimage import feature
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix
from skimage import feature
import numpy as np
from sklearn.model_selection import GridSearchCV

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        
    # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        
    def describe(self, image, eps=1e-7):
        
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        
        # return the histogram of Local Binary Patterns
        return hist


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="path to the training images")
ap.add_argument("-e", "--testing", required=True,  help="path to the tesitng images")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
# loop over the training images
for imagePath in paths.list_images(args["training"]):
    
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist) 

    
# train a SVM model on the data
model1=SVC(random_state=42)
model1.fit(data, labels)
params={'C':[250,255,260,270],'kernel':['linear','rbf'],'gamma':[160,170,180,200,150]}
grid=GridSearchCV(model1, param_grid=params,cv=10)
grid.fit(data,labels)
best=grid.best_params_

model=SVC(C=best['C'],random_state=42,max_iter=5000,kernel=best['kernel'],gamma=best['gamma'])
model.fit(data, labels)
#loop over training images for prediction over data

train_pred=[]
train_labels=[]
for imagePath in paths.list_images(args["training"]):
    
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    train_labels.append(imagePath.split(os.path.sep)[-2])
    train_pred.append(model.predict(hist.reshape(1,-1))[0])
print("accuracy for training is {} :".format(accuracy_score(train_labels,train_pred)))
print(confusion_matrix(train_labels,train_pred))
print(precision_score(train_labels,train_pred,labels=["live","spoof"],pos_label="spoof"))
print(recall_score(train_labels,train_pred,labels=["live","spoof"],pos_label="spoof"))  
data1=[]
test_labels=[]
test_pred=[]
# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    test_labels.append((imagePath.split(os.path.sep)[-2]))
    data1.append(hist)
    test_pred.append((prediction[0]))

print("accuracy of the test set {}:".format(accuracy_score(test_labels,test_pred)))
print(confusion_matrix(test_labels,test_pred))
print("precision for the test set {}:".format(precision_score(test_labels,test_pred,labels=["live","spoof"],pos_label="spoof")))
print("recall for the test set {}:".format(recall_score(test_labels,test_pred,labels=["live","spoof"],pos_label="spoof")))   