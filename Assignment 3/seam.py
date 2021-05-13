
#---------importing the necessary libraries---------------
import sys
import cv2
import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy import ndimage as ndi
from numba import jit



"""
def find_energy_of_pixels(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)
    img = energy_map.astype(np.uint8)
    cv2.imshow('energy ap', img)
    return energy_map
"""










#------calculate the energy of all the pixels of the image-----------
def find_energy_of_pixels(im):
    row, col, channel = im.shape
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    #grad_mag = np.absolute(np.sum(xgrad, axis=2) + np.sum(ygrad, axis=2))
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))
    img = grad_mag.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.imshow('grad',img)
    #plt.imshow(img)            ## for displaying the energy function
    #plt.show()
    return grad_mag

#------finding the minimum cummulative energy path from top to bottom------------
@numba.jit
def find_minimum_seam(img):
    row, col, channel = img.shape
    energy = find_energy_of_pixels(img)
    CME = np.copy(energy)
    backtrack = np.zeros_like(CME, dtype=int)       # backtrack for getting the seam from top to bottom or left to right
    for i in range(1,row):
        for j in range(0,col):
            if j==0:
                index = np.argmin(CME[i-1,j:j+2])
                backtrack[i,j] = j+index
                min_energy = CME[i,j+index]
            else:
                index = np.argmin(CME[i-1,j-1:j+2])
                backtrack[i,j] = j-1+index
                min_energy = CME[i,j-1+index]
            CME[i,j] += min_energy;
    return CME, backtrack


    
#------removing a single column pixels from the image with minimum energy-----------
@numba.jit
def remove_single_column(img,image=None):
    row, col, channel = img.shape
    CME, backtrack = find_minimum_seam(img)
    mask = np.ones((row,col), dtype = bool)
    j = np.argmin(CME[-1])
    img = show_seam(img,backtrack,j)
    if(image is not None):
        cv2.imshow(image,img)
    for i in reversed(range(row)):
        mask[i,j] = False
        j = backtrack[i,j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((row,col-1,3))
    return img
    
#-----------iterating for removing k columns from the given image---------------
def remove_columns(img,k):
    row, col, channel = img.shape
    for i in range(k):
        img = remove_single_column(img)
    return img

def remove_single_row(img,image):
    img = np.rot90(img,1,(0,1))
    #cv2.imshow('roate',img)
    img = remove_single_column(img,image)
    img = np.rot90(img,3,(0,1))
    return img

   
def show_seam(img,backtrack,j):
    row,col,channel = img.shape
    for i in reversed(range(row)):
        img[i][j][0] = 0
        img[i][j][1] = 0
        img[i][j][2] = 255
        j = backtrack[i,j]
    return img

def show_seam_add(img,seam_index):
    row,col,channel = img.shape
    for i in range(row):
        j = seam_index[i]
        img[i][j][0] = 0
        img[i][j][1] = 0
        img[i][j][2] = 255
    return img
    
@numba.jit
def add_single_column(img,seam_index):
    row,col,channel = img.shape
    ref_img = np.copy(img)
    output_img = np.zeros((row,col+1,3))
    seam_index.reverse()
    for i in range(row):
        j = seam_index[i]
        for k in range(3):
            if j==0:
                avg_energy = np.average(ref_img[i,j:j+2,k])
                output_img[i,j,k] = ref_img[i,j,k]
                output_img[i,j+1,k] = avg_energy
                output_img[i,j+2:,k] = ref_img[i,j+1:,k]
            else:
                avg_energy = np.average(ref_img[i,j-1:j+1,k])
                output_img[i,:j,k] = ref_img[i,:j,k]
                output_img[i,j,k] = avg_energy
                output_img[i,j+1:,k] = ref_img[i,j:,k]
    output_img = output_img.astype(np.uint8)
    show_seam_add(output_img,seam_index)
    #cv2.imshow('output image',output_img)
    return output_img

def add_columns(img,k):
    ref_img = np.copy(img)
    final_img = np.copy(img)
    row,col,channel = img.shape
    all_seam_indexes = []
    for i in range(k):
        CME, backtrack = find_minimum_seam(ref_img)
        j = np.argmin(CME[-1])
        seam_index = form_seam_index(backtrack,j,row)
        all_seam_indexes.append(seam_index)
        ref_img = remove_single_column(ref_img,'ref_images')
    for i in reversed(range(k)):
        seam_index = all_seam_indexes[i]
        img = add_single_column(img,seam_index)
    return img
        
def add_rows(img,k):
    img = np.rot90(img,1,(0,1))
    img = add_columns(img,k)
    img = np.rot90(img,3,(0,1))
    return img
    
    
def form_seam_index(backtrack,j,row):
    seam_index = []
    for i in reversed(range(row)):
        seam_index.append(j)
        j = backtrack[i,j]
    return seam_index

def remove_seam_grayscale(im, backtrack):
    row, col = im.shape[:2]
    return im[backtrack].reshape((row, col - 1))


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def object_removal(img, rmask, horizontal_removal=False):
    img = img.astype(np.float64)
    rmask = rmask.astype(np.float64)
    output = img

    row, col = img.shape[:2]

    while len(np.where(rmask > 10)[0]) > 0:
        seam_index, backtrack = get_minimum_seam(output,rmask)
        if(horizontal_removal):
            output = remove_single_row(output)
            rmask = rotate_image(rmask,True)
            rmask = remove_seam_grayscale(rmask,backtrack)
            rmask = rotate_image(rmask,False)
        else:
            output = remove_single_column(output)
            rmask = remove_single_column(rmask)
            

    num_add = (row if horizontal_removal else col) - output.shape[1]
    #output, mask = seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)
    if(horizontal_removal):
        output = add_rows(output,num_add)
    else:
        output = add_columns(output,num_add)
    
                                 

    return output      
    















#-----------------main function-------------------

def main():
    img = cv2.imread('river.png')
    cv2.imshow('original',img)
    col_img = np.copy(img)
    row_img = np.copy(img)
    col_added = np.copy(img)
    row_added = np.copy(img)
    i=0
    j=0
    col = 30
    row = 50
    """
    while(i<col):
        key = cv2.waitKey(1) & 0xFF
        if key==ord("c"):
            break
        col_img = remove_single_column(col_img,'column image')
        i+=1
    print(col_img.shape)
    """
    """
    while(j<row):
        key = cv2.waitKey(1) & 0xFF
        if key==ord("c"):
            break
        row_img = remove_single_row(row_img,'row image')
        j+=1
    """
    
    #col_added = add_columns(col_added,50)
    #cv2.imshow('cols added',col_added)
    row_added = add_rows(row_added,30)
    cv2.imshow('rows',row_added)
    #(row_added.shape)
    
if __name__ == '__main__':
    main()
