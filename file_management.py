
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


# creates the matrix A for eigenvals to calculate later
# also saves the mean for later use
# dir, the path to the database
def create_A(dir):
    # create the matrix a transposed
    a = []
    for face in glob.glob(dir + '/persons/*/faces/*.npy'):
        a.append(np.load(face))

    # calculates mean between rows and saves
    m = np.mean(a, axis=0)
    np.save(dir + '/mean', m)
    
    # substracts to each row the mean and returns transposed
    return np.transpose(np.subtract(a, m))



