import library as l
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-i", "--image", required=True, help="path to the image to check")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="confidence ratio")
ap.add_argument("-k", "--kpca", default=False, action='store_true', help="use kpca instead of pca")
args = vars(ap.parse_args())

# get faces
faces = l.extract_face(args['path'], args['image'], args['confidence'])

# show all faces and choose one
print('Look all the faces from the images and select one')
for face in faces:
    l.show_face(face)
num = int(input('Insert the index of the face to search: ')) - 1

# recognize the ohm image
ohm_img = l.get_ohm_image(faces[num], args['path'], args['kpca'])

# searches for the index of the matching face
(i, err) = l.face_space_distance(ohm_img, args['path'])
print('Error is: ' + str(err))

# gets the corresponding path given the index
path = l.get_matching_path(i, args['path'])
print(path)
