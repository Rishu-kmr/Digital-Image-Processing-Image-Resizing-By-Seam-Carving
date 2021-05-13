#-------importing the necessary libraries---------------
import sys
from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import cv2
import numba

#---------function to find the minimum energy of all the pixel values of the given image-----------
def calculate_energy_of_pixels(img):
    dx = np.array([                         # dx derivative of the given pixel using sobel filter
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
        ])
    dx = np.stack([dx] * 3, axis=2)       # converting it to 3d to be convolved with the given image

    dy = np.array([                         # dy derivative of the given pixel using sobel filter
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])
    dy = np.stack([dy] * 3, axis=2)       # converting it to 3d to be convolved with the given image
    
    #img = img.astype('float32')
    #using the partial derivative to calculate the energy value
    convolved = np.absolute(convolve(img, dx)) + np.absolute(convolve(img, dy))
    #s = convolved.astype(np.uint8)
    cv2.imshow('convolved',convolved)
    # sum all the values of the 3 channels
    energy_map = convolved.sum(axis=2)
    return energy_map

def crop_c(img, cols):
    r, c, _ = img.shape
    #new_c = int(c-cols)
    for i in trange(cols):
        img = carve_column(img)
    return img

def crop_r(img, rows):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, rows)
    img = np.rot90(img, 3, (0, 1))
    return img

def add_c(img,k):
    r,c,_ = img.shape
    for i in trane(range(k)):
        img = engrave_column(img)
    return img



@numba.jit
def carve_column(img):
    r, c, _ = img.shape

    CME, backtrack = minimum_seam(img)
    mask = np.ones((r,c), dtype = bool)
    j = np.argmin(CME[-1])
    for i in reversed(range(r)):
        mask[i,j] = False
        j = backtrack[i, j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

@numba.jit
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calculate_energy_of_pixels(img)

    CME = energy_map.copy()
    backtrack = np.zeros_like(CME, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(CME[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = CME[i-1, idx + j]
            else:
                idx = np.argmin(CME[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = CME[i - 1, idx + j - 1]

            CME[i, j] += min_energy
    return CME, backtrack

def show_minimum_seam(img):
    r, c, _ = img.shape
    CME, backtrack = minimum_seam(img)
    line_img = np.copy(img)
    k = np.argmin(CME[-1])
    for i in reversed(range(r)):
        line_img[i][k][0] = 0
        line_img[i][k][1] = 0
        line_img[i][k][2] = 255
        k = backtrack[i,k]
    return line_img
    
def main():
    img = cv2.imread('boat.png')
    row = int(input("Enter the number of rows to be removed "))
    col = int(input("Enter the number of columns to be removed "))
    cv2.imshow('image',img)
    imgn = calculate_energy_of_pixels(img)
    imgn = imgn.astype(np.uint8)
    cv2.imshow('energy map',imgn)
    seam_imgy = show_minimum_seam(img)
    seam_imgx = np.rot90(img,1,(0,1))
    seam_imgx = show_minimum_seam(seam_imgx)
    seam_imgx = np.rot90(seam_imgx,3,(0,1))
    cv2.imshow('seam image y',seam_imgy)
    cv2.imshow('seam image x',seam_imgx)
    """
    img1 = crop_r(img,row)
    img2 = crop_c(img,col)
    cv2.imshow('row',img1)
    cv2.imshow('col',img2)
    """
    
if __name__ == '__main__':
    main()
