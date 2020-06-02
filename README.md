# Fingerprint-Spoof-Detector-based-on-LBP

# Implementation:

1.	Conversion of each image into grayscale before we extract the LBP features.
2.	Extraction of LBP features from the LocalBinaryPattern implementation found in scikit-image.
3.	SVC is used as it tries to classify the classes based on maximum margin by taking extreme points.
4.	Performed GridSearch on SVC to find out that non-linear kernel -RBF perform well when compared to the linear kernel.
5.	Best parameters fitted to our model.
6.	We can see the result our model based on our selected performance metrics.

# Performance Metrics:
1. Accuracy
2. Precision
3. Recall
4. Confusion matrix

# Programming/Libraries:
1. Python
2. opencv
3. sklearn

# References:
1. https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
2. https://scikit-learn.org/
