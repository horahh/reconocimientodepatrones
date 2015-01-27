#!/usr/bin/python
# -*- coding: utf-8 -*-
import urllib2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os,sys

def euclidean_distance(a,b):
    return abs(a-b)

######################################################################
dir = "yalefaces/yalefaces/"
files = os.listdir(dir)
#print files
normal_files = [ elem for elem in files if "normal" in elem]
glasses_files = [ elem for elem in files if "glasses" in elem]
sad_files = [ elem for elem in files if "sad" in elem]

normal_files.sort()
glasses_files.sort()
sad_files.sort()
print normal_files
print glasses_files
print sad_files

######################################################################

## Leemos la imagen como un numpy array
kk = plt.imread(dir+normal_files[0])
m,n = kk.shape[0:2] #get the size of the images
print "img size = %d, %d" % (m,n)

normal_images = np.array([np.array(plt.imread(dir+normal_files[i]).flatten()) for i in range(len(normal_files))],'f')
glasses_images = np.array([np.array(plt.imread(dir+glasses_files[i]).flatten()) for i in range(len(glasses_files))],'f')
sad_images = np.array([np.array(plt.imread(dir+sad_files[i]).flatten()) for i in range(len(sad_files))],'f')

matrix = normal_images

## Leemos la imagen desde la url
#components = (20,40)
components = [20]
for i in components:
    ## Nos quedamos con i componentes principales
    pca = PCA(n_components = i)
    ## Ajustamos para reducir las dimensiones
    reduced_matrix = pca.fit_transform(matrix)

    x,y  = reduced_matrix.shape[0:2]
    print "matriz proyectada"
    print x
    print y
    ## 'Deshacemos' y dibujamos
    reconstructed_matrix  = pca.inverse_transform(reduced_matrix)
    orig_img = reconstructed_matrix[6].reshape(m,n)
    plt.imshow(orig_img, cmap=plt.cm.Greys_r)
    plt.title(u'nº de PCs = %s' % str(i))
    plt.show()

    # aqui se proyecta ya sean las imagenes con anteojos o tristes
    projected_matrix = pca.transform(glasses_images)
    x,y  = projected_matrix.shape[0:2]

    distance = np.array([[ euclidean_distance(projected_matrix[j],reduced_matrix[i]) for i in range(15) ] for j in range(15)])
    #print distance
    c = [ "b", "g", "r","m","c","y","k","b", "g", "r","m","c","y","k","b"]
    for i in range(15):
        plt.plot(range(15), distance[i], c[i]) 
    plt.show()
    #plt.plot(range(15),distance[6],"b")

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



