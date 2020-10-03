import library as l
import tests as t
import test2 as t2
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-n", "--nval", required=True, type=int, help="heuristic n data")
ap.add_argument("-k", "--kpca", default=False, action='store_true', help="use kpca instead of pca")
args = vars(ap.parse_args())

# create the matrix A from the data
A = l.create_A(args['path'])

# calculate and saves eigen values and vectors
# (u2, v2) = l.calculate_eigen(A, args['path'])
# (u, v) = l.manual_eigen(A, args['path'], args['kval'])
# (u3, v3) = t.eigen_values_and_vectors(A, args['path'])
# print("\nManual:\n", u, v)
# print("\Auto:\n", u2[:,:args['kval']], v2[:args['kval']])
# print("\nDif\n", abs(u2[:,:args['kval']]) - abs(u), abs(v2[:args['kval']]) - abs(v))

if args['kpca'] == False:
    # calculate and saves eigen values and vectors
    # (u1, v1) = l.manual_eigen(A, args['path'], args['nval'])
    (u2, v2) = l.calculate_pca_eigen(A, args['path'])

    # creates and saves the ohm space
    l.create_ohm_space(A, u2, args['path'])
else:
    K = l.create_K(A)
    
    # calculate and saves eigen values and vectors
    (u, v) = l.calculate_kpca_eigen(K, args['path'])

    # creates and saves the ohm space
    l.create_ohm_space(K, u, args['path'], True)
