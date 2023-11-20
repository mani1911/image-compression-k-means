import sys
from skimage import io
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]

# Y = [95404286.98810115, 187264628.1696772, 277250438.35128206, 363357797.3740872, 449195469.93302774, 534162601.46219414, 611476493.6256964, 694383401.9162117, 768040874.6659051]
# X =  [1, 2, 3, 4, 5, 6, 7, 8, 9]

Y = []
X = []

# reading the image
image = io.imread(filename)
rows, cols = image.shape[0], image.shape[1]
img_original = image
image = image.reshape(rows * cols, 3)
points = 64 * 64 * 3

def findInertia(image, centroids):
    inertia = 0
    for i in range(640):
         for j in range(640):
              if image[i][j] not in centroids:
                   for c in centroids:
                        inertia += np.linalg.norm(image[i][j] - c)
    return inertia

def findKMeans(image):

    for i in range(1,10):
        kMeans = KMeans(n_clusters = i)
        kMeans.fit(image)
        centers = np.asarray(kMeans.cluster_centers_, dtype=np.uint8)

        # print(centers)

        inertia =  findInertia(image=img_original, centroids=centers)

        Y.append(inertia / points)
        X.append(i)

def plot(nums1, nums2):
    plt.plot(nums1, nums2)
    plt.savefig('test')
     

findKMeans(image)
print(X, Y)
# print(X, Y)
# plot(X, Y)


    