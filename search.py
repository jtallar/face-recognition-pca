import library as l

dir = './data'
face = []

# recognize the ohm image
ohm_img = l.get_ohm_image(face, dir)
print("Ohm img, ", ohm_img)

# searches for the index of the matching face
i = l.face_space_distance(ohm_img, dir)
print(i)

# gets the corresponding path given the index
path = l.get_matching_path(i, dir)
print(path)