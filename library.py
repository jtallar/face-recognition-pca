
# imports used packages
from pathlib import Path
import numpy as np
import os
import glob

# saves the image given as a matrix in the correspondig dir
# face is a matrix of N x N 
# name is the name of the person on the image
# dir is the database path
def save_face(face, name, dir):

    # creates the directory if the path does not exist
    path = os.path.join(dir, name, 'faces')
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # counts the files on the path
    files_count = len(os.listdir(path))
	
    # saves the matrix as a vector
    np.save(path + str(files_count), face.flatten())


##################################################################


# creates the matrix A for eigenvals to calculate later
# also saves the mean for later use
# dir, the path to the database
# returns (A, index), with index being the dirs of all the indexes
def create_A(dir):
    # create the matrix a transposed
    a = []
    index = []
    for face in glob.glob(dir + '/persons/*/faces/*.npy'):
        a.append(np.load(face))
        index.append(face)
    
    # calculates mean between rows and saves
    m = np.mean(a, axis=0)
    np.save(dir + '/mean', m)
    
    # substracts to each row the mean and returns transposed
    return (np.transpose(np.subtract(a, m)), index)


# calculates the eigenvalues and eigenvectos given a matrix
# returns touple (u, s), being u eigenvectors and s the eigenvalues
# TODO to be replaced with our function
def calculate_eigen(a, dir):
    u, s, vh = np.linalg.svd(np.dot(np.transpose(a), a), full_matrices=True)
    # u --> eigenvectors in 3 x 3 matrix, s --> eigenvalues in vector
   
    u = np.dot(a, u)
    for i in range(0, len(u[0])):
        u[:,i] = np.transpose(np.array(u[:,i]/np.linalg.norm(u[:,i], 2)))
    # u --> eigenvector matrix of the covariance, with ||u||=1
    # s --> eigenvalues in vector

    # saves to files the eigen values and vectors
    np.save(dir + '/eigenvector', u)
    np.save(dir + '/eigenvalues', s)
    return (u, s)


# creates and saces the ohm space
# a the A matrix N^2 x M
# u the eigenvectors N^2 x K
# dir the dir to save
# returns the ohm space
def create_ohm_space(a, u, dir):
    o = []

    # for each col get the weights
    for i in range(0, len(a[0])):
        ohmi = np.dot(np.transpose(u), np.transpose(a)[i])
        o.append(ohmi)

    # calculate ohm space and save
    res = np.transpose(o)
    np.save(dir + '/ohm-space', res)
    return res


######################################################################


# ohm1 is a K vector with K weights
# ohm2 is a K vector with K weights
# eigenvalues is a K vector with K eigenvalues
# returns face space distance between ohm1 and ohm2
def vector_distance(ohm1, ohm2, eigenvalues):
    if (len(ohm1) != len(ohm2)):
        raise Exception("Vectors must have the same lengths")
    sum = 0
    for i in range(0, len(ohm1)):
        sum += (ohm1[i] - ohm2[i]) ** 2 / eigenvalues[i]
    return sum

# ohm is a K vector with K weights from a particular image
# ohm_space is a K x M matrix with each individual ohm vector
#  in each column
# threshold is the max distance to recognize image
# returns image number recognized (minimum vector distance) 
#  or -1 if unknown
def face_space_distance(ohm_img, ohm_space, eigenvalues, threshold=float('Inf')):
    index = -1
    min_err = float('Inf')
    for i in range(0, len(ohm_space[0])):
        dist = vector_distance(ohm_img, ohm_space[:,i], eigenvalues)
        if (dist < threshold and dist < min_err):
            index = i
            min_err = dist
    print("Min error ",min_err)
    return index