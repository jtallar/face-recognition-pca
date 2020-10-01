import library as l
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-i", "--image", required=True, help="path to the image to check")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="confidence ratio")
args = vars(ap.parse_args())

# get faces
faces = l.extract_face(args['path'], args['image'], args['confidence'])

# save faces
for face in faces:
    l.show_face(face)
    name = input("Enter face name: ")
    l.save_face(face, name, args['path'])