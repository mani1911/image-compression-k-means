import sys
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import os
import shutil

from kneed import KneeLocator, DataGenerator

dataset_path = 'dataset/'
output_path = 'output/'

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Y = [7661231898.168653, 3290707373.3764653, 2266018605.274069, 1512373450.479423, 1196348509.879937, 966578851.923748, 822065746.9691927, 721009283.7247559, 627830528.0580585, 566890254.0646206, 520692339.94309807]

# Find optimal k value by plotting elbow curve 
def FindKFromImage(imagePath):
    image = io.imread(imagePath)
    rows, cols = image.shape[0], image.shape[1]
    image = image.reshape(rows * cols, 3)
    X = []
    Y = []
    for i in range(1, 12):  
        kMeans = KMeans(n_clusters = i)
        kMeans.fit(image)
        X.append(i)
        Y.append(kMeans.inertia_)
    kneedle = KneeLocator(X, Y, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    return optimal_k

# create the dataset for the model
def CreateDataset():
    os.makedirs('output', exist_ok=True)
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            elbowK = str(FindKFromImage(file_path))
            os.makedirs(f'output/{elbowK}', exist_ok=True)

            destination_path = os.path.join(output_path + elbowK, filename)
            shutil.copy(file_path, destination_path)

    


# FindKFromImage('dataset/me.jpeg')
CreateDataset()
