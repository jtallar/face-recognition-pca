import library as l

# function to call on calculate button
def calculate(path, kpca, nval):
    # create the matrix A from the data
    A = l.create_A(path)

    # calculate and saves eigen values and vectors
    # (u, v) = l.calculate_pca_eigen(A, path, nval)
    # (u2, v2) = l.calculate_pca_eigen_auto(A, path, nval)
    # print("\nManual:\n", u, v)
    # print("\Auto:\n", u2, v2)
    # print("\nDif\n", abs(u2) - abs(u), abs(v2) - abs(v))

    if kpca == False:
        # calculate and saves eigen values and vectors
        (u, v) = l.calculate_pca_eigen(A, path, nval)

        # creates and saves the ohm space
        l.create_ohm_space(A, u, path)
    else:
        K = l.create_K(A)
        
        # calculate and saves eigen values and vectors
        (u, v) = l.calculate_kpca_eigen(K, path, nval)

        # creates and saves the ohm space
        l.create_ohm_space(K, u, path, True)