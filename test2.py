import numpy as np

def customSVD(matrix):
    matrix = np.matrix(matrix)
    transposed = np.transpose(matrix)
    aux_matrix = np.matmul(matrix, transposed)
    eigenvalues, eigenvectors = customEigenCalc(aux_matrix)
    eigenvectors = matrix.transpose() * eigenvectors
    for i in range(0, matrix.shape[0]):
        eigenvectors[:, i] = eigenvectors[:, i] / norm_2(eigenvectors[:, i])
    return eigenvectors.transpose()


def customEigenCalc(matrix):
    Q, R = qr_decomposition(matrix)
    eigvectors = Q

    A = np.matmul(R, Q)
    error = 0.00001
    maxiterations = 200
    i = 0
    while i < maxiterations:
        Q, R = qr_decomposition(A)
        A = np.matmul(R, Q)
        eigvectors = np.matmul(eigvectors, Q)
        i += 1
    return readEigenvalues(R), np.matrix(eigvectors)


def readEigenvalues(A):
    eigenvalues = []
    error = 0.00001
    for i in range(0, A.shape[0]):
        eigenvalue = A.item((i, i))
        if abs(eigenvalue) <= error:
            eigenvalue = 0
        eigenvalues.append(eigenvalue)
    return np.array(eigenvalues)


# checks for lower triangles of val < admissibleError in matrix
def isConvergingTriangular(matrix, admissibleError):
    for x in range(0, matrix.shape[0]):
        for y in range(0, matrix.shape[1]):
            if x != y and abs(matrix[x, y]) > admissibleError:
                return False
    return True


def qr_decomposition(matrix):
    A = np.array(matrix)
    Q = np.matrix([[]])
    for i in range(0, A.shape[0]):
        u = A[:][i]
        for j in range(0, i):
            u = np.subtract(u, np.dot(A[:][i], Q[:][j]) * Q[j])
        a = u / np.linalg.norm(u)
        if i == 0:
            Q = np.array([a])
        else:
            Q = np.vstack((Q, a))
    Q = np.transpose(Q)
    R = np.zeros(A.shape)
    for i in range(0, A.shape[0]):
        for j in range(i, A.shape[1]):
            R[i, j] = np.dot(Q[:, i], A[:][j])
    return Q, R


def norm_2(vec):
    vec = np.array(vec)
    return np.sqrt(sum([i ** 2 for i in vec]))


def testing():
    matrix = [[0, 3, 1], [5, 9, 1], [1, 8, 3], [5, 6, 6]]
    matrix = np.matrix(matrix).transpose()
    print("Matrix oridinal: ")
    # print(matrix)
    # print(np.transpose(matrix))
    #   exit(0)
    #   aux = customSVD([matrix])
    print(np.array(matrix))
    print('--')
    aux = customSVD(np.array(matrix))

    #   aux = isConvergingTriangular([ [1, 1, 1],
    #                                  [0, 0, 1],
    #                                  [0, 0, 1]], 0.00001 )
    print("Eigenvalues: " + str(aux))
    caca = np.linalg.svd(np.matrix(matrix), full_matrices=False)
    print(caca)

    eigvec = customSVD(matrix)
    print(eigvec)