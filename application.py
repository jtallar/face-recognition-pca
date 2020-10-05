from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
#from db import Database
from PIL import Image, ImageTk
import glob
import tkinter as tk
import library as l
import os

DATABASE_PATH = "/Users/geronimomaspero/Desktop/mna-tpe1/data"
INITIAL_DIR = "/Users/geronimomaspero/Desktop/mna-tpe1"


#db = Database('store.db')




#---------------------# FUNCTIONS #---------------------#

# def populate_list():
#     parts_list.delete(0, END)
#     for row in db.fetch():
#         parts_list.insert(END, row)


# def add_item():
#     if part_text.get() == '' or customer_text.get() == '' or retailer_text.get() == '' or price_text.get() == '':
#         messagebox.showerror('Required Fields', 'Please include all fields')
#         return
#     db.insert(part_text.get(), customer_text.get(),
#               retailer_text.get(), price_text.get())
#     parts_list.delete(0, END)
#     parts_list.insert(END, (part_text.get(), customer_text.get(),
#                             retailer_text.get(), price_text.get()))
#     clear_text()
#     populate_list()


# def select_item(event):
#     try:
#         global selected_item
#         index = parts_list.curselection()[0]
#         selected_item = parts_list.get(index)

#         part_entry.delete(0, END)
#         part_entry.insert(END, selected_item[1])
#         customer_entry.delete(0, END)
#         customer_entry.insert(END, selected_item[2])
#         retailer_entry.delete(0, END)
#         retailer_entry.insert(END, selected_item[3])
#         price_entry.delete(0, END)
#         price_entry.insert(END, selected_item[4])
#     except IndexError:
#         pass


# def remove_item():
#     db.remove(selected_item[0])
#     clear_text()
#     populate_list()


# def update_item():
#     db.update(selected_item[0], part_text.get(), customer_text.get(),
#               retailer_text.get(), price_text.get())
#     populate_list()


# def clear_text():
#     part_entry.delete(0, END)
#     customer_entry.delete(0, END)
#     retailer_entry.delete(0, END)
#     price_entry.delete(0, END)


#
# -------------------------------------- CODIGO PROBETA -----------------------------------------------------------------------------------
#
def popupresults(face):
    #Build windows
    popup2 = tk.Toplevel(app)
    popup2.wm_title("Face recognition results")
    popup2.geometry('400x100+600+480')

    # LLAMAR A LA FUNCION QUE NOS DA LOS RESULTADOS

    # recognize the ohm image
    ohm_img = l.get_ohm_image(face, DATABASE_PATH)

    # searches for the index of the matching face
    (i, err) = l.face_space_distance(ohm_img, DATABASE_PATH)
    # Error Label
    label = Label(popup2, text='Error is: ' + str(err), font=('bold', 14))
    label.grid(row=0, column=2)
    # gets the corresponding path given the index
    path = l.get_matching_path(i, DATABASE_PATH)

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
    image_path = filedialog.askopenfilename( initialdir=INITIAL_DIR, title="Select a File", filetypes=( ("All Files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")))
    my_image = ImageTk.PhotoImage(Image.open(image_path))
    #creo un label y a ese label le cargo la imagen
    my_image_label = Label(image=my_image)
    #Ubico a mi label en pantalla
    my_image_label.place(height=400, width=400,x=400,y=200)

    # get faces
    faces = l.extract_face(DATABASE_PATH, image_path, 0.2)

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
    path = filedialog.askdirectory( initialdir=INITIAL_DIR, title="Select a File")
    #this is a list with all the images' paths.
    list_of_items = glob.glob(path + '/*.jpeg')
    for file in list_of_items:
        #Hasta aca tenemos un ciclo por todas las imagenes que el usuario quiere subir

        # get faces
        faces = l.extract_face(DATABASE_PATH, file, 0.2)

        # save faces
        for face in faces:
            #l.show_face(face)
            #Popeamos una ventanita que le pida nombre para la imagen y la meta en la scroll list.
            name = popupmessage(face)
            l.save_face(face, name.get(), DATABASE_PATH)

    #Calculamos todo
    # create the matrix A from the data
    A = l.create_A(DATABASE_PATH)

    # calculate and saves eigen values and vectors
    (u, v) = l.calculate_eigen(A, DATABASE_PATH )

    # creates and saves the ohm space
    l.create_ohm_space(A, u, DATABASE_PATH)

        


    





#---------------------# FUNCTIONS #---------------------#

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












# # Retailer
# retailer_text = StringVar()
# retailer_label = Label(app, text='Retailer', font=('bold', 14))
# retailer_label.grid(row=1, column=0, sticky=W)
# retailer_entry = Entry(app, textvariable=retailer_text)
# retailer_entry.grid(row=1, column=1)

# # Price
# price_text = StringVar()
# price_label = Label(app, text='Price', font=('bold', 14))
# price_label.grid(row=1, column=2, sticky=W)
# price_entry = Entry(app, textvariable=price_text)
# price_entry.grid(row=1, column=3)

#SCROLL
# # Parts List (Listbox)
# parts_list = Listbox(app, height=16, width=50)
# parts_list.grid(row=3, column=0, columnspan=3, rowspan=6, pady=20, padx=20)

# # Create scrollbar
# scrollbar = Scrollbar(app)
# scrollbar.grid(row=3, column=3)

# # Set scroll to listbox
# parts_list.configure(yscrollcommand=scrollbar.set)
# scrollbar.configure(command=parts_list.yview)

# # Bind select
# parts_list.bind('<<ListboxSelect>>', select_item)



# Buttons
# add_btn = Button(app, text='Add Part', width=12, command=add_item)
# add_btn.grid(row=2, column=0, pady=20)

# remove_btn = Button(app, text='Remove Part', width=12, command=remove_item)
# remove_btn.grid(row=2, column=1)

# update_btn = Button(app, text='Update Part', width=12, command=update_item)
# update_btn.grid(row=2, column=2)

# clear_btn = Button(app, text='Clear Input', width=12, command=clear_text)
# clear_btn.grid(row=2, column=3)



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