import library as l
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-k", "--kval", required=True, type=int, help="heuristic k data")
args = vars(ap.parse_args())

# create the matrix A from the data
A = l.create_A(args['path'])

# calculate and saves eigen values and vectors
# (u1, v1) = l.manual_eigen(A, args['path'], args['kval'])
(u2, v2) = l.calculate_eigen(A, args['path'])

# creates and saves the ohm space
l.create_ohm_space(A, u2, args['path'])
