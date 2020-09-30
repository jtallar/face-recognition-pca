import numpy as np

import rref

A = np.mat("3 2; 1 0")
B = np.mat("1 2 3; 4 5 6; 7 4 9")

# print("A: \n", A)
print("B: \n", B)

# print("Eigenvalues: \n", np.linalg.eigvals(A))

eigenvalue,eigenvector = np.linalg.eig(B)

# print("First tuple of eig:\n", eigenvalue)
# print("Second tuple of eig:\n", eigenvector)

# print("Charpoly of B: \n", np.poly(B))
roots = np.roots(np.poly(B))
print("Roots of charpoly: \n", roots)
print("\nReal Eigenvalues of B: \n", np.diag(eigenvalue))
print("Manual Eigenvalues of B: \n", np.diag(roots))






## ESTO DE ABAJO SON PRUEBAS AUN, EL DEF RREF CREO QUE ANDA 






### Create an RREF instance with your matrix.
# r = rref.RREF(B)

# ### Print the matrix to check results.
# result = r.mm.matrix
# print([i for i in result])


# def rref(B, tol=1e-8, debug=False):
#   A = B.copy()
#   rows, cols = A.shape
#   r = 0
#   pivots_pos = []
#   row_exchanges = np.arange(rows)
#   for c in range(cols):

#     ## Find the pivot row:
#     pivot = np.argmax (np.abs (A[r:rows,c])) + r
#     m = np.abs(A[pivot, c])
#     if m <= tol:
#       ## Skip column c, making sure the approximately zero terms are
#       ## actually zero.
#       A[r:rows, c] = np.zeros(rows-r)
#     else:
#       ## keep track of bound variables
#       pivots_pos.append((r,c))

#       if pivot != r:
#         ## Swap current row and pivot row
#         A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
#         row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]
        
#       ## Normalize pivot row
#       A[r, c:cols] = A[r, c:cols] / A[r, c]

#       ## Eliminate the current column
#       v = A[r, c:cols]
#       ## Above (before row r):
#       if r > 0:
#         ridx_above = np.arange(r)
#         A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
#       ## Below (after row r):
#       if r < rows-1:
#         ridx_below = np.arange(r+1,rows)
#         A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
#       r += 1
#     ## Check if done
#     if r == rows:
#       break
#   return (A, pivots_pos, row_exchanges)

# A, pivots, row = rref(B)
# print("A = ", A)
# print("pivots = ", pivots)
# print("row = ", row)