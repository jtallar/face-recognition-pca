import library as l
import tests as t
import test2 as t2
import argparse
import numpy as np
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-k", "--kval", required=True, type=int, help="heuristic k data")
args = vars(ap.parse_args())

# create the matrix A from the data
A = l.create_A(args['path'])

# calculate and saves eigen values and vectors
# (u1, v1) = l.manual_eigen(A, args['path'], args['kval'])
# (u2, v2) = l.calculate_eigen(A, args['path'])
# (u3, v3) = t.eigen_values_and_vectors(A, args['path'])

u2, s, vh = np.linalg.svd(np.dot(np.transpose([[0, 3, 1], [5, 9, 1], [1, 8, 3]]), full_matrices=True)
u4 = t2.customSVD(A)

# print("\nManual:\n", u1)
# print("\nEigenvalues ->\nAuto:\n", v2,"\nMagic:\n", v3)
print("\nEigenvectors ->\nAuto: \n", u2)
print("\nMagic: \n", u4)

# print("Magic: ",len(u3),"x", len(u3[0]),"\tAuto:", len(u2),"x", len(u2[0]))
# creates and saves the ohm space
# l.create_ohm_space(A, u4, args['path'])
