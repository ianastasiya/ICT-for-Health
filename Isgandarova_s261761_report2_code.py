# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:59:21 2020

@author: isken
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



#N1 Open file and check original image
filein='low_risk_1.jpg'
im = mpimg.imread(filein)#im is an Ndarray with shape 583*583*3
plt.figure()
plt.imshow(im)
plt.title('Original image')
plt.pause(0.1)

#N2 reshape the image to 2D from 3D
N1, N2, N3= im.shape # N3=3, the number of elementary colors, i.e. red, green ,blue
# im(i,j,1) stores the amount of red color from 0 to 255
# im(i,j,2) stores the amount of green color from 0 to 255
# im(i,j,3) stores the amount of blue color from 0 to 255
# we resize the original image
im_2D=im.reshape((N1*N2,N3))# N1*N2 rows and N3 columns
# im_2D is a sequence of colors, that can take 2^24 different values
Nr,Nc=im_2D.shape

#get a image with only Ncluster colors
Ncluster=3# number of clusters/quantized colors we want to have in the simpified image:
kmeans = KMeans(n_clusters=Ncluster, random_state=0)# instantiate the object K-means:
# run K-means:
kmeans.fit(im_2D)
kmeans_centroids=kmeans.cluster_centers_.astype('uint8')# get the centroids (i.e. the 3 colors). Note that the centroids are float numbers,but we need uint8 numbers to show the image
# copy im_2D into im_2D_quant
im_2D_quant=im_2D.copy()
for kc in range(Ncluster):
    quant_color_kc=kmeans_centroids[kc,:]
    # kmeans.labels_ stores the cluster index for each of the Nr pixels
    # find the indexes of the pixels that belong to cluster kc
    ind=(kmeans.labels_==kc)
    # set the quantized color to these pixels
    im_2D_quant[ind,:]=quant_color_kc
im_quant=im_2D_quant.reshape((N1,N2,N3))
plt.figure()
plt.imshow(im_quant,interpolation=None)
plt.title('Image with quantized colors')
#plt.draw()
plt.pause(0.1)

#Clustering is finished, then we will work on finding contour of darkest cluster and evaluate area of cluster and so on
#Step 1 - find the contour of the darkest cluster, corresponding to the mole
centroids=kmeans_centroids
sc=np.sum(centroids,axis=1)
i_col=sc.argmin()# index of the cluster that corresponds to the darkest color
# Step 2 -  define the 2D-array where in position i,j you have the number of the cluster pixel i,j belongs to 
im_clust=kmeans.labels_.reshape(N1,N2)
#plt.matshow(im_clust)
# Step 3 - find the positions i,j where im_clust is equal to i_col
zpos=np.argwhere(im_clust==i_col)
# Step 4 -  ask the user to write the number of objects belonging to
# cluster i_col in the image with quantized colors
N_spots_str=input("How many distinct dark spots can you see in the image? ")
N_spots=int(N_spots_str)
# Step 5 -  find the center of the mole
if N_spots==1:
    center_mole=np.median(zpos,axis=0).astype(int)
else:
    # use K-means to get the N_spots clusters of zpos
    kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
    kmeans2.fit(zpos)
    centers=kmeans2.cluster_centers_.astype(int)
    # the mole is in the middle of the picture:
    center_image=np.array([N1//2,N2//2])
    center_image.shape=(1,2)
    d=np.zeros((N_spots,1))
    for k in range(N_spots):
        d[k]=np.linalg.norm(center_image-centers[k,:])
    center_mole=centers[d.argmin(),:]    
# Step 6 -  take a subset of the image that includes the mole
c0=center_mole[0]
c1=center_mole[1]
RR,CC=im_clust.shape
stepmax=min([c0,RR-c0,c1,CC-c1])
cond=True
area_old=0
surf_old=1
step=10# each time the algorithm increases the area by 2*step pixels 
# horizontally and vertically
im_sel=(im_clust==i_col)# im_sel is a boolean NDarray with N1 rows and N2 columns
im_sel=im_sel*1# im_sel is now an integer NDarray with N1 rows and N2 columns
while cond:
    subset=im_sel[c0-step:c0+step+1,c1-step:c1+step+1]
    area=np.sum(subset)
    Delta=np.size(subset)-surf_old
    surf_old=np.size(subset)
    if area>area_old+0.01*Delta:
        step=step+10
        area_old=area
        cond=True
        if step>stepmax:
            cond=False
    else:
        cond=False
        # subset is the serach area
plt.matshow(subset)

dim = np.shape(subset)

subset[0,:] = 0
subset[dim[0]-1,:] = 0
subset[:,0] = 0
subset[:,dim[1]-1] = 0

#fill the holes inside the mole
for i in range(dim[0]-1):
    for j in range(dim[1]-1):
        one_left = np.where(subset[i,0:j] == 1)
        one_right = np.where(subset[i,j+1:dim[0]-1] == 1)
        dim_ol = np.shape(one_left)
        dim_or = np.shape(one_right)
        if dim_ol[1]>0 and dim_or[1]>0:
            one_up = np.where(subset[0:i,j] == 1)
            one_down = np.where(subset[i+1:dim[1]-1,j] == 1)
            dim_ou = np.shape(one_up)
            dim_od = np.shape(one_down)
            if dim_ou[1]>0 and dim_od[1]>0:
                subset[i,j] = 1

#remove external spots

for i in range(0,round(dim[0]/2))[::-1]: 
        for j in range(round(dim[1]/2),dim[1]-1): 
            ng = np.sum(subset[i-1,j-1:j+1] + subset[i+1,j-1:j+1] + subset[i,j-1] + subset[i,j+1])
            if subset[i,j] == 1 and ng<=4: 
               subset[i,j] = 0
        for j in range(0,round(dim[1]/2))[::-1]: 
            ng = np.sum(subset[i-1,j-1:j+1] + subset[i+1,j-1:j+1] + subset[i,j-1] + subset[i,j+1])
            if subset[i,j] == 1 and ng<=4: 
                subset[i,j] = 0
for i in range(round(dim[0]/2),dim[0]-1): 
        for j in range(round(dim[1]/2),dim[1]-1): 
            ng = np.sum(subset[i-1,j-1:j+1] + subset[i+1,j-1:j+1] + subset[i,j-1] + subset[i,j+1])
            if subset[i,j] == 1 and ng<=4: 
                subset[i,j] = 0
        for j in range(0,round(dim[1]/2))[::-1]: 
            ng = np.sum(subset[i-1,j-1:j+1] + subset[i+1,j-1:j+1] + subset[i,j-1] + subset[i,j+1])
            if subset[i,j] == 1 and ng<=4: 
                subset[i,j] = 0
#show cleaned image
plt.figure(figsize=(30,40))
plt.matshow(subset)
plt.show()

#evaluate area and corresponding circumference
area_final = np.sum(subset)
print("Area of the mole: ", area_final)
radius = np.sqrt(area_final/np.pi)
circle = 2 * np.pi * radius

#retain only 1 next to at least a zero to display contour
contour = subset.copy()
for i in range(dim[0]-1): 
	for j in range(dim[1]-1): 
		ng = np.sum(subset[i,j-1] + subset[i,j+1] + subset[i-1,j] + subset[i+1,j])
		if subset[i,j] == 1 and ng==4:
			contour[i,j] = 0

plt.figure(figsize=(30,40))
plt.matshow(contour)
plt.show()

##evaluate perimeter
perimeter = np.copy(subset)
for i in range(dim[0]-1): 
     for j in range(dim[1]-1): 

           if subset[i][j] == 1 and subset[i-1][j] == 1 and subset[i+1][j] == 1 and subset[i][j-1] == 1 and subset[i][j+1] == 1:
                perimeter[i][j] = 0

Perimeter_mole = 0
for i in range(dim[0]-1):
    for j in range(dim[1]-1):
                    if perimeter[i][j] == 1:
                        Perimeter_mole += 1

print("Perimeter of the mole: ", Perimeter_mole)


#evaluate ratio
ratio_sqrt_2 = Perimeter_mole/circle
print("Ratio between perimeter and circumference: ", ratio_sqrt_2)


y, x = np.nonzero(contour)
#subtract mean from each dimension
x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])
#print(coords)
#Covariance matrix and its eigenvectors and eigenvalues
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)
sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]
#plot principal components
scale = 20
plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')
plt.plot(x, y, 'k.')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
plt.show()