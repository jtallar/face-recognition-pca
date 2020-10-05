from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
#from db import Database
from PIL import Image, ImageTk
import glob
import tkinter as tk
import library as l
import os
import argparse


# Read arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to data")
ap.add_argument("-n", "--nval", required=True, type=int, help="heuristic n data")
ap.add_argument("-k", "--kpca", default=False, action='store_true', help="use kpca instead of pca")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="confidence ratio")
args = vars(ap.parse_args())


def popupresults(face):
    #Build windows
    popup2 = tk.Toplevel(app)
    popup2.wm_title("Face recognition results")
    popup2.geometry('400x100+600+480')

    # LLAMAR A LA FUNCION QUE NOS DA LOS RESULTADOS

    # recognize the ohm image
    ohm_img = l.get_ohm_image(face, args['path'], args['kpca'])

    # searches for the index of the matching face
    (i, err) = l.face_space_distance(ohm_img, args['path'])
    # Error Label
    label = Label(popup2, text='Error is: ' + str(err), font=('bold', 14))
    label.grid(row=0, column=2)
    # gets the corresponding path given the index
    path = l.get_matching_path(i, args['path'])

    path2 = os.path.dirname(path)
    # Path Label
    label = Label(popup2, text= os.path.basename(path2), font=('bold', 14))
    label.grid(row=1, column=2)
    
    # IMPRIMIR LOS RESULTADOS 

    # Display save button
    button_save = Button(popup2, text="Next", command=popup2.destroy)
    button_save.grid(row=3, column=2)

    #when this popup is done waiting for user's input, it will return
    popup2.wait_window()

        
def open_image():
    global my_image
    image_path = filedialog.askopenfilename( initialdir=os.getcwd(), title="Select a File", filetypes=( ("All Files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")))
    if image_path:
        my_image = ImageTk.PhotoImage(Image.open(image_path))
        #creo un label y a ese label le cargo la imagen
        my_image_label = Label(image=my_image)
        #Ubico a mi label en pantalla
        my_image_label.place(height=400, width=400,x=400,y=200)

        # get faces
        faces = l.extract_face(args['path'], image_path, args['confidence'])

        # Now we iterate in all the faces
        for face in faces:
            l.show_face(face)
            #Popeamos una ventanita que le pida nombre para la imagen y la meta en la scroll list.
            name = popupresults(face)      

        
def popupmessage(face):
    #Build windows
    popup = tk.Toplevel(app)
    popup.wm_title("Name Assignment")
    popup.geometry('400x100+600+480')

    # Display Label
    label = Label(popup, text="Please assign the name of the person above", font=('bold', 14))
    label.grid(row=0, column=2)

    # Obtain image from the path_variable (argument) 
    #global my_image2
    #my_image2 = ImageTk.PhotoImage(Image.open(path_variable))
    
    # Display image in the popup
    #my_image_label2 = Label(popup,image=my_image2)
    #my_image_label2.grid(row=1, column=2)

    l.show_face(face)

    # Display input box
    user_text = StringVar()
    user_entry = Entry(popup, textvariable=user_text)
    user_entry.grid(row=2, column=2)

    # Display save button
    button_save = Button(popup, text="Save", command=popup.destroy)
    button_save.grid(row=3, column=2)

    #when this popup is done waiting for user's input, it will return
    popup.wait_window()
    return user_text


def open_directory():
    path = filedialog.askdirectory( initialdir=os.getcwd(), title="Select a File")
    #this is a list with all the images' paths.
    list_of_items = glob.glob(path + '/*.jpeg')
    for file in list_of_items:
        #Hasta aca tenemos un ciclo por todas las imagenes que el usuario quiere subir

        # get faces
        faces = l.extract_face(args['path'], file, args['confidence'])

        # save faces
        for face in faces:
            #l.show_face(face)
            #Popeamos una ventanita que le pida nombre para la imagen y la meta en la scroll list.
            name = popupmessage(face)
            l.save_face(face, name.get(), args['path'])

    l.process_data(args['path'], args['nval'], args['kpca'])       



#---------------------# APPLICATION #---------------------#    

# Create window object
app = tk.Tk()

#---------------------# FRONT END #---------------------#

#Quit button
button_quit = Button(app, text="Exit Program", command=app.quit)
button_quit.place(height=50, width=600, x=100, y=730 )

# UPLOAD DATA TITLE
title_upload_data = Label(app, text='UPLOAD DATA', font=('bold', 14), pady=20)
title_upload_data.place(height=50, width=400, x=0, y=0)

# RECOGNIZE FACE TITLE
title_recognize_face = Label(app, text='RECOGNIZE FACE', font=('bold', 14), pady=20)
title_recognize_face.place(height=50, width=400, x=400, y=0)


# Choose Directory Text - for UPLOAD DATA section
upload_image_label = Label(app, text='Select folder with all your images', font=('bold', 14), pady=20)
upload_image_label.place(height=50, width=400, x=0, y=70)

# Choose Directory Button - for UPLOAD DATA section
upload_image_button = Button(app, text="Select folder...", command=open_directory)
upload_image_button.place(height=50, width=400, x=0, y=140)

# Upload Image Text - for RECOGNIZE FACE section
upload_image_label2 = Label(app, text='Upload Image', font=('bold', 14))
upload_image_label2.place(height=50, width=400, x=400, y=70)


# Upload Image Button - for RECOGNIZE FACE section
upload_image_button2 = Button(app, text="Open Image...", command=open_image)
upload_image_button2.place(height=50, width=400, x=400, y=140)

app.title('Face Recognition - Probeta Technologies')
#app['background']='#DBF3FA'
app.geometry('800x800')

# Populate data
#populate_list()

# Start program
app.mainloop()


# To create an executable, install pyinstaller and run
# '''
# pyinstaller --onefile --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' part_manager.py
# '''

#---------------------# FRONT END #---------------------#