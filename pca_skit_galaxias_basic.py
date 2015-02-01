# -*- coding: utf-8 -*-
import urllib2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os,sys
from skimage.color import rgb2gray
from skimage import io, filter
from skimage import img_as_float
from skimage.transform import rotate


######################################################################
dir = "all/"
files = os.listdir(dir)
#print files
eliptic_files = [ elem for elem in files if "eliptica" in elem]
spiral_files = [ elem for elem in files if "spiral" in elem]
all_files = [ elem for elem in files ]


# remove the train images
#sa = ( set( all ) - set(spiral) ) #- set( eliptic)

#all = list( sa)



eliptic_files.sort()
spiral_files.sort()
all_files.sort()

print eliptic_files 
print spiral_files
print all_files

######################################################################

# join all references for learning matrix
ref_files = eliptic_files + spiral_files

print ref_files
print all_files


## Leemos la imagen como un numpy array
kk = plt.imread(dir+ref_files[0])
m,n = kk.shape[0:2] #get the size of the images
print "img size = %d, %d" % (m,n)

ref_images = np.array([np.array(plt.imread(dir+ref_files[i]).flatten()) for i in range(len(ref_files))],'f')
#eliptic_images = np.array([np.array(plt.imread(dir+eliptic_files[i]).flatten()) for i in range(len(eliptic_files))],'f')
#spiral_images = np.array([np.array(plt.imread(dir+spiral_files[i]).flatten()) for i in range(len(spiral_files))],'f')
all_images = np.array([np.array(plt.imread(dir+all_files[i]).flatten()) for i in range(len(all_files))],'f')

#all_images=ref_images

matrix = ref_images

orig_img = ref_images[0].reshape(m,n,3)
print "min val:" , np.amin(orig_img)
print "max val:" , np.amax(orig_img)
plt.imshow(orig_img, cmap=plt.cm.Greys_r)
plt.show()


img_gray = rgb2gray(orig_img.astype(np.uint8))
plt.imshow(img_gray, cmap=plt.cm.Greys_r)
plt.show()


edges = filter.sobel(img_gray)
io.imshow(edges)
io.show()

img_rot = rotate(edges,30)
plt.imshow(img_rot, cmap=plt.cm.Greys_r)
plt.show()

#plt.imshow(img_gray, cmap=plt.cm.Greys_r)
plt.title(u'nº de PCs = %s' % str(i))
plt.show()

print "matrix dimentions"
for i in ref_images:
    print len(i)

## Leemos la imagen desde la url
#components = (20,40)
components = [40] 
all_size = len(all_images)
ref_size = len(ref_images)
for components in [40]:
    ## Nos quedamos con i componentes principales
    pca = PCA(n_components = components)
    ## Ajustamos para reducir las dimensiones
    x,y  = matrix.shape[0:2]
    print "original matrix len: (%d,%d)" % (x, y)
    reduced_matrix = pca.fit_transform(matrix)
    print len(reduced_matrix[0])

    print "matriz proyectada con set original:"
    x,y  = reduced_matrix.shape[0:2]
    print "reduced matrix dimentions: %d, %d" % ( x, y )
    ## 'Deshacemos' y dibujamos
    reconstructed_matrix  = pca.inverse_transform(reduced_matrix)
    print "reconstructed matrix (array) dimention"
    print len(reconstructed_matrix[0])
    orig_img = reconstructed_matrix[0].reshape(m*3,n)
    plt.imshow(orig_img, cmap=plt.cm.Greys_r)
    plt.title(u'nº de PCs = %s' % str(components))
    plt.show()

    projected_matrix = pca.transform(all_images)
    x,y  = projected_matrix.shape[0:2]
    print "matriz proyectada con set prueba:"
    print "reduced matrix dimentions: %d, %d" % ( x, y )

    distance = np.array([[ np.linalg.norm(projected_matrix[j]-reduced_matrix[i]) for i in range(ref_size) ] for j in range(all_size)])
    print distance
    c = [ "b", "g", "r","m"] #,"c","y","k","b", "g", "r","m","c","y","k","b"]
    #for i in range(all_size):
    for i in range(all_size):
        file_twin = np.argmin(distance[i])
        plt.title(u'%s, nº de PCs = %s\n%s' % (all_files[i], str(components), ref_files[file_twin]))
        plt.plot(range(ref_size), distance[i], c[i%4]) 
        plt.show()

print "matriz reducida"
x,y  = kk.shape[0:2]
print x
print y
print np.linalg.norm(kk.T[0])

pca = PCA()
pca.fit(matrix)
# cortamos para ver los primeros 50 valores solamente
# que corresponden a los valores mas significativos
varianza = pca.explained_variance_ratio_[:50]
var_acum= np.cumsum(varianza)
plt.plot(range(len(varianza)), varianza,"r")
plt.plot(range(len(varianza)), var_acum,"b")
plt.show()



