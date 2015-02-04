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
from skimage.filter import gaussian_filter


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

# join all references for learning matrix
ref_files = eliptic_files + spiral_files

# Prints list of files or not
def print_image_list(print_list):
    if print_list:
        print "\nFiles lists:"
        print
        print eliptic_files
        print
        print spiral_files
        print
        print all_files
        print
        print       ref_files
        print
        print all_files
    return

print_image_list(0)

## Leemos la imagen como un numpy array
kk = plt.imread(dir+ref_files[0])
m,n = kk.shape[0:2] #get the size of the images
print "img size = %d, %d" % (m,n)

## Definiciones de tipos de imagenes
ref_images = np.array([np.array(plt.imread(dir+ref_files[i]).flatten()) for i in range(len(ref_files))],'f')
#eliptic_images = np.array([np.array(plt.imread(dir+eliptic_files[i]).flatten()) for i in range(len(eliptic_files))],'f')
#spiral_images = np.array([np.array(plt.imread(dir+spiral_files[i]).flatten()) for i in range(len(spiral_files))],'f')
all_images = np.array([np.array(plt.imread(dir+all_files[i]).flatten()) for i in range(len(all_files))],'f')

#all_images=ref_images

matrix = ref_images
def restore_image(flatten_image):
    orig_img = flatten_image.reshape(m,n,3)
    #print "min val:" , np.amin(orig_img)
    #print "max val:" , np.amax(orig_img)
    plt.imshow(orig_img, cmap=plt.cm.Greys_r)
    #plt.show()
    return orig_img

#orig_img = restore_image(ref_images[13])

def convert_to_gray(image):
    img_gray = rgb2gray(image.astype(np.uint8))
    #plt.imshow(img_gray, cmap=plt.cm.Greys_r)
    #plt.show()
    return img_gray

# gray_img = convert_to_gray(orig_img)
# plt.imshow(gray_img, cmap=plt.cm.Greys_r)
# plt.title(u'Gray Image' )
# plt.show()

# #edges = filter.sobel(img_gray)
# #io.imshow(edges)
# #io.show()

def convert_to_binary(gray_image):
    blurred_image = gaussian_filter(gray_image,sigma=9)
    return np.where(blurred_image > np.mean(blurred_image),1.0,0.0)

# ## Binary Image
# binary_image = convert_to_binary(gray_img)
# io.imshow(binary_image)
# io.show()

# plt.title(u'Graph of overlapping Areas' )
# plt.plot(range(len(binary_image)), np.sum(binary_image,axis=0), 'r') 
# plt.plot(range(len(binary_image)), np.sum(binary_image,axis=1), 'g') 
# plt.show()

def boundary_find(binary_image):
    x_dist = np.sum(binary_image,axis=0)
    y_dist = np.sum(binary_image,axis=1)

    img_center = len(binary_image)/2
    for i in range(img_center):
        x_right_limit = img_center+i 
        if x_dist[x_right_limit] < 20 :
            break
    
    for i in range(img_center):
        x_left_limit = img_center-i 
        if x_dist[x_left_limit] < 20 :
            break

    for i in range(img_center):
        y_right_limit = img_center+i 
        if y_dist[y_right_limit] < 20 :
            break
    
    for i in range(img_center):
        y_left_limit = img_center-i 
        if y_dist[y_left_limit] < 20 :
            break
    return (x_left_limit,x_right_limit,y_left_limit,y_right_limit)

# black_coordinates = boundary_find(binary_image)

def black_boundary(bin_image,coordinate):
    for x in range(len(bin_image)):
# doesnt matter for an square image to take x rather than y
        for y in range(len(bin_image)):
            if( y < coordinate[0] or 
                y > coordinate[1] or 
                x < coordinate[2] or 
                x > coordinate[3]):
                bin_image[x][y] = 0;
    return bin_image

# filter_image = black_boundary(binary_image,black_coordinates)

# ##Filtered Image
# io.imshow(filter_image)
# io.show()

def apply_mask(img, mask):
    for x in range(len(img)):
        for y in range(len(img)):
            if(mask[x][y] == 1):
                img[x][y] = img[x][y]
            else:
                img[x][y] = 0
    return

# apply_mask(gray_img,filter_image)

# ##Gray Image after mask
# io.imshow(gray_img)
# io.show()
    
def get_cov(bin_img):
    coord_list =[]
    for x in range(len(bin_img)):
        for y in range(len(bin_img)):
            if bin_img[x][y]:
                coord_list.append([x,y])

    coord_vector = np.array(coord_list)
    return np.cov(coord_vector[:,0],coord_vector[:,1])

# cov_mat = get_cov(filter_image)

def rotate_image(gray_image,cov_mat,view):
    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    max_eigen = np.argmax(eig_val_cov)
    if view:
        print "max eigen" , max_eigen
    rot_angle = np.arctan(eig_vec_cov[max_eigen][0]/eig_vec_cov[max_eigen][1])
    if view:
        print eig_vec_cov[max_eigen][1]
        print eig_vec_cov[max_eigen][0]
    rot_angle = np.degrees(rot_angle)
    if view:
        print "rot angle", rot_angle
    return rotate(gray_image,-rot_angle)

# rot_img = rotate_image(gray_img,cov_mat,1)


# plt.imshow(rot_img, cmap=plt.cm.Greys_r)
# plt.title(u'Gray Image Aligned to Axis')
# plt.show()

# print "matrix dimensions"
# for i in ref_images:
#     print len(i)

## PROCESS ALL IMAGES WITH ALL FILTERS
# Pass 1 to view plots or 0 to hide them
def process_images_filters(ref_images,view):
    print "\nProcessing images through all filters:\n"
    for i in range(len(ref_images)):
        #Original Image, restore since we flattened it
        original_img = restore_image(ref_images[i])
        #Gray Image
        # io.imshow(gray_img)
        # io.show()

    	gray_image = convert_to_gray(original_img)
    	if view:
           plt.imshow(gray_image, cmap=plt.cm.Greys_r)
    	   plt.title('Unfiltered Gray Scale : Image # %d' %i)
    	   plt.show()
           plt.clf()
        #Binary Image
        binary_image = convert_to_binary(gray_image)
        if view:
            io.imshow(binary_image)
            io.show()
        # Overlapping areas
            plt.title(u'Graph of overlapping Areas : Image # %d' %i )
            plt.plot(range(len(binary_image)), np.sum(binary_image,axis=0), 'r') 
            plt.plot(range(len(binary_image)), np.sum(binary_image,axis=1), 'g') 
            plt.show()
        #Find black coordinates
        black_coordinates = boundary_find(binary_image)
        #Filtered Image
        filter_image = black_boundary(binary_image, black_coordinates)
        if view:
            io.imshow(filter_image)
            io.show()
        #Applied Mask
        apply_mask(gray_image,filter_image)
        if view:
            io.imshow(gray_image)
            io.show()
        #Covariance Matrix    
        cov_mat = get_cov(filter_image)
        #Rotated image at adequate axis
        rot_img = rotate_image(gray_image,cov_mat,0)
        if view:
            plt.imshow(rot_img, cmap=plt.cm.Greys_r)
            plt.title(u'Filtered Gray Scale Aligned to Axis : Image # %d' %i)
            plt.show()
        ##Save Figures
        #fig = plt.figure()
        #fig.savefig('saved/figure_' + '%d.jpg' %i)
        #Flatten images to prepare them for PCA Analysis
        # TODO -- FLATTENED IMAGE ARRAY 
        #ref_images_filtered = np.array([np.array(rot_img.flatten())],f])         
    return #ref_images_filtered

def process_images_to_grayscale(all_images,view):
    print "Processing images to grayscale:\n"
    for i in range(len(all_images)):
        #Original Image, restore since we flattened it
        original_img = restore_image(all_images[i])
        # Gray Image
        gray_image = convert_to_gray(original_img)
        if view:
            io.imshow(gray_image)
            io.show()
        # TODO -- GRAYSCALE FLATTENNED ARRAY
        #inner_array = np.array(gray_image.flatten())    
    #all_images_grayscale = np.array(inner_array,'f') 
    return #all_images_grayscale


## ---------------  IMAGE PRE-PROCESSING CALLS ---------------  ## 

#Call to process reference images through all filters: grayscale, binary, mask and axis rotation
process_images_filters(matrix,0)

#Process all available image sets to grayscale
process_images_to_grayscale(all_images,0)

## Leemos la imagen como un numpy array
#kk = plt.imread(dir+ref_files[0])
#m,n = kk.shape[0:2] #get the size of the images
#ref_images = np.array([np.array(plt.imread(dir+ref_files[i]).flatten()) for i in range(len(ref_files))],'f')

## ---------------  PCA ANALYSIS OF FILTERERED & GRAYSCALE IMAGES ---------------  ## 
## Leemos la imagen desde la url
#components = (20,40)
# #COMENTO DESDE ACA PARA ABAJO
# components = [40] 
# all_size = len(all_images)
# ref_size = len(ref_images)
# for components in [40]:
#     ## Nos quedamos con i componentes principales
#     pca = PCA(n_components = components)
#     ## Ajustamos para reducir las dimensiones
#     x,y  = matrix.shape[0:2]
#     print "original matrix len: (%d,%d)" % (x, y)
#     reduced_matrix = pca.fit_transform(matrix)
#     print len(reduced_matrix[0])

#     print "matriz proyectada con set original:"
#     x,y  = reduced_matrix.shape[0:2]
#     print "reduced matrix dimentions: %d, %d" % ( x, y )
#     ## 'Deshacemos' y dibujamos   
#     reconstructed_matrix  = pca.inverse_transform(reduced_matrix)
#     print "reconstructed matrix (array) dimension"
#     pr        if view:int len(reconstructed_matrix[0])
#     orig_img =    econstructed_matrix[0].reshape(m*3,n)
    #     plt           mshow(orig_img, cmap=plt.cm.Greys_r)
#     plt.title(u'nº de PCs = %s' % str(components))
#     plt.show()

#     projected_matrix = pca.transform(all_images)
#     x,y  = projected_matrix.shape[0:2]
#     print "matriz proyectada con set prueba:"
#     print "reduced matrix dimensions: %d, %d" % ( x, y )

#     distance = np.array([[ np.linalg.norm(projected_matrix[j]-reduced_matrix[i]) for i in range(ref_size) ] for j in range(all_size)])
#     print distance
#     c = [ "b", "g", "r","m"] #,"c","y","k","b", "g", "r","m","c","y","k","b"]
#     #for i in range(all_size):
#     for i in range(all_size):
#         file_twin = np.argmin(distance[i])
#         plt.title(u'%s, nº de PCs = %s\n%s' % (all_files[i], str(components), ref_files[file_twin]))
#         plt.plot(range(ref_size), distance[i], c[i%4]) 
#         plt.show()

# print "matriz reducida"
# x,y  = kk.shape[0:2]
# print x
# print y
# print np.linalg.norm(kk.T[0])

# pca = PCA()
# pca.fit(matrix)
# # cortamos para ver los primeros 50 valores solamente
# # que corresponden a los valores mas significativos
# varianza = pca.explained_variance_ratio_[:50]
# var_acum= np.cumsum(varianza)
# plt.plot(range(len(varianza)), varianza,"r")
# plt.plot(range(len(varianza)), var_acum,"b")
# plt.show()




