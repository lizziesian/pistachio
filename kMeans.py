from spectral import *
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np

#Opens the hyperspectral cube after PCA
img = envi.open('postPCAcube.hdr', 'postPCAcube.dat')

#Crops the image to half of the initial dimensions
img = img[400:1600, 400:1200, :]

img_width = img.shape[0]
img_height = img.shape[1]

#Runs the kmeans algorithm from the spectral module 
#Using 10 clusters for a maximum of 100 iterations
(classmap, centroids) = kmeans(img, 10, 1000)

#Flattens the map of labels for each pixel
cluster_labels = classmap.flatten()

#Sets the same 10 colours for each colour map
cluster_colours = [[0.06292562, 0.81650338, 0.52821701],
                   [0.25632692, 0.3738871,  0.20280424],
                   [0.54238974, 0.69872758, 0.92536983],
                   [0.39337304, 0.83885786, 0.27255063],
                   [0.49426522, 0.39722835, 0.95811467],
                   [0.50504622, 0.52819654, 0.36604863],
                   [0.97205696, 0.90535343, 0.22825141],
                   [0.46368176, 0.78006571, 0.04962603],
                   [0.91118215, 0.65378407, 0.58648027],
                   [0.11872315, 0.91446523,0.45501632]]

#Assigns each pixel a colour based on its label following k-means
segmented = np.zeros((img_width, img_height, 3))
for i in range(img_width):
    for j in range(img_height):
        label = cluster_labels[i * img_height + j]
        segmented[i, j] = cluster_colours[label]

#Count the occurences of each colour
unique, counts = np.unique(segmented.flatten(), return_counts=True)
print (dict(zip(unique, counts)))

#Plot the k-means clustered colour map
plt.figure(figsize=(10,10))
imshow(segmented)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colours[label], label=str(label + 1)) for label in range(10)], bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Segmented Pistachio Image using K-means')
plt.savefig('labelmapPostPCAcube.png')


#Plots the centroids of each cluster at each iteration
#plt.show()
# for i in range(centroids.shape[0]):
#     plt.plot(centroids[i])
# plt.grid()
# plt.show()
