import numpy as np

def rref(B, tol=1e-8):
  A = B.copy()
  rows, cols = A.shape
  r = 0
  pivots_pos = []
  row_exchanges = np.arange(rows)
  for c in range(cols):

    ## Find the pivot row:
    pivot = np.argmax (np.abs (A[r:rows,c])) + r
    m = np.abs(A[pivot, c])
    if m <= tol:
      ## Skip column c, making sure the approximately zero terms are
      ## actually zero.
      A[r:rows, c] = np.zeros(rows-r)
    else:
      ## keep track of bound variables
      pivots_pos.append((r,c))

      if pivot != r:
        ## Swap current row and pivot row
        A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
        row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]
        
      ## Normalize pivot row
      A[r, c:cols] = A[r, c:cols] / A[r, c]

      ## Eliminate the current column
      v = A[r, c:cols]
      ## Above (before row r):
      if r > 0:
        ridx_above = np.arange(r)
        A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
      ## Below (after row r):
      if r < rows-1:
        ridx_below = np.arange(r+1,rows)
        A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
      r += 1
    ## Check if done
    if r == rows:
      break
  return A

A = np.mat("4 1 1; 2 -1 0; 1 2 0")
print("A: \n", A)
eigenvalue,eigenvector = np.linalg.eig(A)

# print("Charpoly of B: \n", np.poly(A))
roots = np.roots(np.poly(A))
# print("Roots of charpoly: \n", roots)
print("\nReal Eigenvalues of A: \n", np.diag(eigenvalue))
print("\nManual Eigenvalues of A: \n", np.diag(roots))
    
print("\nReal Eigenvectors of A: \n", eigenvector)

v = []
N = A[0].size
for i in range(N):
    si = roots[i]
    Atilde = (A - si * np.identity(N))
    Atilde_red = rref(Atilde)
    out1 = -Atilde_red[:, N-1][0].tolist()[0][0]
    out2 = -Atilde_red[:, N-1][1].tolist()[0][0]
    res = np.array([out1, out2, 1])
    res = res / np.linalg.norm(res)
    v = np.concatenate((v, res), axis=0)

# print(v)
v = v.reshape((N,N))
v = np.transpose(v)
print("\nManual Eigenvectors of A: \n", v)

# s1 = roots[0]
# Atilde = (A - s1 * np.identity(3))
# Atilde_red = rref(Atilde)
# print("\nAtilde_red: \n",Atilde_red)
# out1 = -Atilde_red[:, 2][0].tolist()[0][0]
# out2 = -Atilde_red[:, 2][1].tolist()[0][0]
# res = np.array([out1, out2, 1])
# print(res)




v = []
N = A[0].size
for i in range(N):
    si = roots[i]
    Atilde = (A - si * np.identity(N))
    Atilde_red = rref(Atilde)
    res = []
    for k in range(N-1):
        aux = -Atilde_red[:, N-1][k].tolist()[0][0]
        res[k]=aux
    # out1 = -Atilde_red[:, N-1][0].tolist()[0][0]
    # out2 = -Atilde_red[:, N-1][1].tolist()[0][0]
    res[N] = 1
    res = res / np.linalg.norm(res)
    v = np.concatenate((v, res), axis=0)