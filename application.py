from tkinter import *
import applib as lib
import library as l
import glob

import os
from tkinter import filedialog as Filedialog


from PIL import Image, ImageTk

W_RATIO = 1.5
W_BGCOL ='#494949'
W_FGCOL = 'lightgray'

DIR = 'data'


# generates centered window
# w the window opened
# ratio the window ratio to mantain
# returns window geometry [width]x[height]+[startx]+[starty]
def get_window_geo(w, ratio):
    width = int(w.winfo_screenwidth() / ratio)
    height = int(w.winfo_screenheight() / ratio)
    startx = int((w.winfo_screenwidth() - width) / 2)
    starty = int((w.winfo_screenheight() - height) / 2)
    return "{}x{}+{}+{}".format(width, height, startx, starty)



# set defaults on window app
root = Tk()
root.geometry(get_window_geo(root, W_RATIO))
root.configure(bg=W_BGCOL)
root.title('Face Recognition - Probeta Technologies')

# render basic layout
load_frame = LabelFrame(root, text="Load Faces", bg=W_BGCOL, fg=W_FGCOL)
load_frame.place(relx=0.04, rely=0.04, relheight=0.65, relwidth=0.45)

search_frame = LabelFrame(root, text="Search Faces", bg=W_BGCOL, fg=W_FGCOL)
search_frame.place(relx=0.51, rely=0.04, relheight=0.65, relwidth=0.45)

config_frame = LabelFrame(root, text="Configurations", bg=W_BGCOL, fg=W_FGCOL)
config_frame.place(relx=0.04, rely=0.71, relheight=0.25, relwidth=0.92)


####################### configurations layout #######################

# algorithm selection frame and layout
algorithm = IntVar()

alg_frame = Frame(config_frame, bg=W_BGCOL)
alg_frame.place(relx=0.05, rely=0.2, relheight=0.8, relwidth=0.2)

label = Label(alg_frame, text='Pre-Processing Algorithm', bg=W_BGCOL, fg=W_FGCOL)
label.pack(anchor=CENTER)

pca_btn = Radiobutton(alg_frame, text='PCA', variable=algorithm, value=1, bg=W_BGCOL, fg=W_FGCOL, highlightbackground=W_BGCOL, selectcolor=W_BGCOL)
pca_btn.pack(anchor=W)

kpca_btn = Radiobutton(alg_frame, text='KPCA', variable=algorithm, value=2, bg=W_BGCOL, fg=W_FGCOL, highlightbackground=W_BGCOL, selectcolor=W_BGCOL)
kpca_btn.pack(anchor=W)
algorithm.set(1)

# k selection fram and layout
k_frame = Frame(config_frame, bg=W_BGCOL)
k_frame.place(relx=0.3, rely=0.25, relheight=0.5, relwidth=0.2)

label = Label(k_frame, text='K Value', bg=W_BGCOL, fg=W_FGCOL)
label.pack(anchor=CENTER)

k_value = IntVar()
scale = Scale(k_frame, variable = k_value, resolution=1, orient=HORIZONTAL, from_=0, to=100, bg=W_BGCOL, fg=W_FGCOL, highlightbackground=W_BGCOL)
scale.pack(anchor=CENTER, fill=X)
k_value.set(10)

# confidence factor selection fram and layout
c_frame = Frame(config_frame, bg=W_BGCOL)
c_frame.place(relx=0.55, rely=0.25, relheight=0.5, relwidth=0.2)

label = Label(c_frame, text='Confidence Factor', bg=W_BGCOL, fg=W_FGCOL)
label.pack(anchor=CENTER)

confidence_factor = DoubleVar()
scale = Scale(c_frame, variable=confidence_factor, orient=HORIZONTAL, resolution=0.1, from_=0, to=1, bg=W_BGCOL, fg=W_FGCOL, highlightbackground=W_BGCOL)
scale.pack(anchor=CENTER, fill=X)
confidence_factor.set(0.2)

# calculate button
cal_frame = Frame(config_frame, bg=W_BGCOL)
cal_frame.place(relx=0.80, rely=0.1, relheight=0.8, relwidth=0.15)

calculate_btn = Button(cal_frame, text ="Preprocess Data", relief=RAISED, borderwidth=0, command=lib.calculate(DIR, algorithm == 2, k_value.get()))
calculate_btn.pack(side=LEFT, fill=X)


####################### loads layout #######################


def analize_images():

    # get directory path
    path = Filedialog.askdirectory(initialdir=os.getcwd(), title="Select a Folder or File")
    global load_img

    # forget button and load image label
    load_btn.place_forget()
    image_label.pack(side=TOP)
    image_btn.place(anchor=SE, relx=1, rely=1, relwidth=0.25, relheight=0.1)
    image_entry.place(anchor=SW, relx=0, rely=1, relwidth=0.7, relheight=0.1)


    # calculate max resolution posible for image
    res = int(min(face_frame.winfo_height(), face_frame.winfo_width()) * 0.8)
    
    # get all images paths (names)
    images = glob.glob(path + '/*.jp*g')
    for image in images:
        faces = l.extract_face(DIR, image, confidence_factor.get())

        # after extract get each face
        for face in faces:
            load_img = ImageTk.PhotoImage(Image.fromarray(face).resize((res,res),1))
            image_label.config(image=load_img)
            image_btn.wait_variable(image_btn_var)


face_frame = Frame(load_frame, bg=W_BGCOL)
face_frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

image_label = Label(face_frame)
image_entry = Entry(face_frame)
image_btn_var = IntVar()
image_btn = Button(face_frame, text='Save Face', command=lambda: image_btn_var.set(1))

load_btn = Button(face_frame, text='Select Image Folder...', command=analize_images)
load_btn.place(relx=0.5, rely=0.5, anchor=CENTER)












root.mainloop()