import library as l

dir = './data'

# create the matrix A from the data
A = l.create_A(dir)

# calculate and saves eigen values and vectors
(u, v) = l.calculate_eigen(A, dir)

# creates and saves the ohm space
l.create_ohm_space(A, u, dir)
