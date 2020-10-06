
# imports used packages
from pathlib import Path
import numpy as np
import cv2
import os
import glob
import tensorflow as tf
from tensorflow import keras

# given a a base dir for the model and the image returns
# list of matrix. Each matrix is a face
# dir base path of prototxt and caffemodel
# image_path the image path
# conf the confidence factor
def extract_face(dir, image_path, conf):

    # load serialized model from disk
    net = cv2.dnn.readNetFromCaffe(os.path.join(dir, 'deploy.prototxt'), os.path.join(dir, 'model.caffemodel'))

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    faces = []

    # loop over the detections
    for i in range(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > conf:

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
    
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # extract face from image, sized at 50x50		
            faces.append(cv2.resize(cv2.cvtColor(image[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY), (50,50)))
    
    return faces


# saves the image given as a matrix in the correspondig dir
# face is a matrix of N x N 
# name is the name of the person on the image
# dir is the database path
def save_face(face, name, dir):

    # creates the directory if the path does not exist
    path = Path(os.path.join(dir, 'persons', name))
    path.mkdir(parents=True, exist_ok=True)
    
    # counts the files on the path
    files_count = len(os.listdir(path))
	
    # saves the matrix as a vector
    np.save(os.path.join(path, str(files_count)), face.flatten())


# shows a face given its matrix
# waits for the user to press escape
def show_face(face):
    cv2.imshow("Press ESC to continue", face)
    #cv2.waitKey(0)
    

##################################################################


# creates the matrix A for eigenvals to calculate later
# also saves the mean and index list for later use
# dir, the path to the database
def create_A(dir):
    # create the matrix a transposed
    a = []
    index = []
    for face in glob.glob(dir + '/persons/*/*.npy'):
        a.append(np.load(face))
        index.append(face)
    
    # calculates mean between rows and saves mean and index
    m = np.mean(a, axis=0)
    np.save(os.path.join(dir, 'mean'), m)
    np.save(os.path.join(dir, 'index'), index)
    
    # substracts to each row the mean and returns transposed
    a = np.transpose(np.subtract(a, m))
    np.save(os.path.join(dir, 'a-matrix'), a)
    return a

# Kernel function used to create kernel matrix
# (X^T * Y + 1) ^ p
def kernel_func(x, y):
    p = 2
    return (np.dot(np.transpose(x), y) + 1) ** p

# creates the K matrix for kpca
# recieves A matrix with Fi/Xi elements, A = [X1 X2 ... XM]
# returns K matrix normalized
def create_K(A):
    M = len(A[0])
    # create kernel matrix
    k = kernel_func(A, A)

    # Normalize K matrix
    unoM = np.ones((M, M)) / M
    k = k - np.dot(unoM, k) - np.dot(k, unoM) + np.dot(unoM, np.dot(k, unoM))

    return k

# calculates the eigenvalues and eigenvectos given a matrix
# returns touple (u, s), being u eigenvectors and s the eigenvalues
# TODO to be replaced with our function
def automatic_eigen(a):
    u, s, vh = np.linalg.svd(a, full_matrices=True)
    # u --> eigenvectors in M x M matrix, s --> eigenvalues in vector
    return (u, s)

# calculates the eigenvalues and eigenvectors for the covariance matrix in pca
# transforming the matrix eigen to the covariance eigen
def calculate_pca_eigen(a, dir, n):
    # TODO: CHANGE TO manual_eigen
    (u, s) = automatic_eigen(np.dot(np.transpose(a), a))
    # u --> eigenvectors in M x M matrix, s --> eigenvalues in vector

    u = np.dot(a, u)
    for i in range(0, len(u[0])):
        u[:,i] = np.transpose(np.array(u[:,i]/np.linalg.norm(u[:,i], 2)))
    # u --> eigenvector matrix of the covariance, with ||u||=1
    # s --> eigenvalues in vector

    # FIXME: UNCOMMENT TO TEST K VALUES
    # cov = np.dot(a, np.transpose(a))
    # aux = np.dot(u, (np.diag(s) ** 0.5))
    # for i in range(0, len(u[0])):
    #     b = np.dot(aux[:,0:i], np.transpose(aux[:,0:i]))
    #     sum = 0
    #     for j in range(0, len(u[0])):
    #         sum += b[j,j]/cov[j,j]
    #         # print("Con los primeros ", i + 1, " autovectores, en el param ", j, " hay % info ", b[j,j]/cov[j,j])
    #     print("Con los primeros ", i + 1, " autovectores, promedio de % de info de ", sum / len(u[0]))

    u = u[:,:n]                                                        # get first k columns
    s = s[:n]

    # saves to files the eigen values and vectors
    np.save(os.path.join(dir, 'eigenvector'), u)
    np.save(os.path.join(dir, 'eigenvalues'), s)
    return (u, s)

# calculates the eigenvalues and eigenvectors for the K matrix in kpca
# returns eigenvalues and eigenvectors of the K matrix, not the Covariance matrix
def calculate_kpca_eigen(k, dir, n):
    # TODO: CHANGE TO manual_eigen
    (u, s) = automatic_eigen(k)
    # u --> eigenvectors in M x M matrix, s --> eigenvalues in vector

    # FIXME: UNCOMMENT TO TEST K VALUES
    # cov = np.dot(a, np.transpose(a))
    # aux = np.dot(u, (np.diag(s) ** 0.5))
    # for i in range(0, len(u[0])):
    #     b = np.dot(aux[:,0:i], np.transpose(aux[:,0:i]))
    #     for j in range(0, len(u[0])):
    #         print("Con los primeros ", i + 1, " autovectores, en el param ", j, " hay % info ", b[j,j]/cov[j,j])

    # TODO: VER COMO DA SIN ESTO, porque me achica mucho los autovectores
    for i in range(0, len(u[0])):
        u[:,i] = u[:,i] / (np.linalg.norm(u[:,i], 2) * (np.sqrt(abs(s[i]))))

    u = u[:,:n]                                                        # get first n columns
    s = s[:n]

    # saves to files the eigen values and vectors
    np.save(os.path.join(dir, 'eigenvector'), u)
    np.save(os.path.join(dir, 'eigenvalues'), s / len(k)) # Save eigenvalues of Covariance

    return (u, s)

# TODO: VER QUE ONDA ESE tol
def rref(B, tol=200000):
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

# algorithm that calculates eigenvalues and eigenvectors given matrix a
# return eigenvalues and k eigenvectors 
def manual_eigen(a):
    # TODO: STOP USING roots and poly
    s = np.roots(np.poly(a))                    # roots of characteristic polynom
    N = len(a)                                  # set N as matrix size
    vector = []                    # initialize result vectors matrix
    for i in range(N):

        # get i-th eigenvalue from s AND A' = (A - Lambda i * Id)
        # aux = sympy.Matrix(a - s[i] * np.identity(N)).rref(iszerofunc=lambda x: abs(x)<1e-16)
        # Atilde_red = np.array(aux[0].tolist(), dtype=float)
        Atilde_red = rref(a - s[i] * np.identity(N))               # A'red --> Gauss-Jordan
        res = []
        for j in range(N-1):
            res.append(-Atilde_red[j][N-1])             # build res
        res.append(1)                                   # last value = 1
        res = res / np.linalg.norm(res)                 # vi = vi / ||vi||
        vector.append(res)                              # append on final v
    
    vector = np.transpose(vector)
    return (vector, s)

# fi is an N^2 vector, Fi = ri - Y
# eigenvectors is an N^2 x K, the K best eigenvectors
# returns omh = K vector with K weights
def calculate_weights_pca(fi, eigenvectors):
    return np.dot(np.transpose(eigenvectors), fi)


# creates and saves the ohm space
# a the A matrix N^2 x M in PCA, the K matrix M x M in KPCA
# u the eigenvectors N^2 x K in PCA, M x M in KPCA
# dir the dir to save
# returns the ohm space
def create_ohm_space(a, u, dir, kpca=False):
    if kpca == True:
        # V^T * K*T
        res = np.dot(np.transpose(u), np.transpose(a))
    else:
        o = []
        # for each col get the weights
        for i in range(0, len(a[0])):
            ohmi = calculate_weights_pca(np.transpose(a)[i], u)
            o.append(ohmi)

        # calculate ohm space and save
        res = np.transpose(o)
    np.save(os.path.join(dir, 'ohm-space'), res)
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
# dir the path for the ohm space and eigenvalues
# threshold is the max distance to recognize image
# returns (i, err) being i image number recognized (minimum vector distance) 
#  or -1 if unknown and err the min error
def face_space_distance(ohm_img, dir, threshold=float('Inf')):

    # load ohm space and eigen values
    ohm_space = np.load(os.path.join(dir, 'ohm-space.npy'))
    eigenvalues = np.load(os.path.join(dir, 'eigenvalues.npy'))

    index = -1
    min_err = float('Inf')
    for i in range(0, len(ohm_space[0])):
        dist = vector_distance(ohm_img, ohm_space[:,i], eigenvalues)
        if (dist < threshold and dist < min_err):
            index = i
            min_err = dist
    return (index, min_err)


# given an image and a dir calculates its ohm
# face the matrix of the image
# dir the direction of the mean and eigenvector to load
def get_ohm_image(face, dir, kpca=False):

    # loads mean already calculated and eigenvector
    m = np.load(os.path.join(dir, 'mean.npy'))
    u = np.load(os.path.join(dir, 'eigenvector.npy'))

    # substracts median to face
    r = np.subtract(face.flatten(), m)

    if kpca == True:
        a = np.load(os.path.join(dir, 'a-matrix.npy'))
        if len(u) != len(a[0]):
            raise Exception("Should be using PCA!")
        # V^T * kernel(r, A)
        return np.dot(np.transpose(u), kernel_func(r, a))
    else:
        if len(u) != len(r):
            raise Exception("Should be using KPCA!")
        # returns the calculated weights
        return calculate_weights_pca(r, u)


# loads and gets the path of the index list
# index the index that matched the face
# dir the path to the list of indexes
def get_matching_path(index, dir):

    # loads the index list of the persons
    dir_list = np.load(os.path.join(dir, 'index.npy'))

    # returns the corresponding path
    return dir_list[index]


######################################################################

# make all calculations por pca or kpca and save ohmspace
# path - path to data
# nval - number of eigenvalues to take from M
# kpca - if true, use kpca. Optional
def process_data(path, nval=6, kpca=False):
    # create the matrix A from the data
    A = create_A(path)

    # calculate and saves eigen values and vectors
    # (u, v) = calculate_pca_eigen(A, path, nval)
    # (u2, v2) = calculate_pca_eigen_auto(A, path, nval)
    # print("\nManual:\n", u, v)
    # print("\Auto:\n", u2, v2)
    # print("\nDif\n", abs(u2) - abs(u), abs(v2) - abs(v))

    if kpca == False:
        # calculate and saves eigen values and vectors
        (u, v) = calculate_pca_eigen(A, path, nval)

        # creates and saves the ohm space
        create_ohm_space(A, u, path)
    else:
        K = create_K(A)
        
        # calculate and saves eigen values and vectors
        (u, v) = calculate_kpca_eigen(K, path, nval)

        # creates and saves the ohm space
        create_ohm_space(K, u, path, True)

# TODO: DEBERIAMOS CREAR SIEMPRE EL MODELO? O GUARDARLO
def create_nn_model(eigenfaces, face_labels, people_count):
    model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),  # 128 nodos de aprendizaje
    keras.layers.Dense(people_count)             # cantidad de personas en la bd 
    ])

    model.compile(optimizer ='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # TODO: determinar epoch --> 20 va mejor, al menos con 11 personas x 8 fotos y en PCA
    # callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(eigenfaces, face_labels, epochs=100, callbacks=[callback])
    model.fit(eigenfaces, face_labels, epochs=20, verbose=0)

    probability_model = keras.Sequential([model, 
                                         keras.layers.Softmax()])
    return probability_model

def get_max_prediction(eigenfaces, face_labels, input, people_count):
    # A partir de las eigenfaces y una imagen de entrada, determinar a qué persona pertenece la imagen de entrada

    # (eigenfaces, face_labels) 
    # (test_image, test_label)

    probability_model = create_nn_model(eigenfaces, face_labels, people_count)
    
    input = (np.expand_dims(input,0))
    
    predictions = probability_model.predict(input)
    
    max_prob = np.argmax(predictions[0])

    return (max_prob, predictions[0][max_prob])


def classify(ohm_img, dir, threshold=float('Inf')):

    # load ohm space and eigen values
    ohm_space = np.load(os.path.join(dir, 'ohm-space.npy'))

    label_list = np.load(os.path.join(dir, 'index.npy'))

    # extract dirname from label_list
    index = 0
    for i in label_list:
        label_list[index] = os.path.dirname(i)
        index += 1
    
    unique_labels = np.unique(label_list)
    m = np.zeros(len(label_list))
    for i in range(0, len(label_list)):
        m[i] = np.where(unique_labels == label_list[i])[0]
    
    (index, prob) = get_max_prediction(np.transpose(ohm_space), m, ohm_img, len(unique_labels))
    return (unique_labels[index], prob)

    