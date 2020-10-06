from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog as Filedialog
import library as l
import glob
import os



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

calculate_btn = Button(cal_frame, text ="Preprocess Data", relief=RAISED, borderwidth=0, command=l.calculate(DIR, algorithm.get() == 2, k_value.get()))
calculate_btn.pack(side=LEFT, fill=X)


####################### loads layout #######################

# analizes and saves images and bla
def analize_images():

    # get directory path
    path = Filedialog.askdirectory(initialdir=os.getcwd(), title="Select a Folder or File")
  
    # get all images paths (names)
    images = glob.glob(path + '/*.jp*g')

    # do only when images is not empty
    if (len(images) != 0):

        # forget button and load image label
        load_btn.place_forget()

        image_label = Label(face_frame)
        image_label.pack(side=TOP)

        image_ety_var = StringVar()
        image_entry = Entry(face_frame, textvariable=image_ety_var)
        image_entry.place(anchor=W, relx=0, rely=0.915, relwidth=0.7, relheight=0.07)

        image_btn_var = IntVar()
        image_btn = Button(face_frame, text='Save Face', command=lambda: image_btn_var.set(1))
        image_btn.place(anchor=SE, relx=1, rely=.90, relwidth=0.25, relheight=0.07)

        image_btn_stop = Button(face_frame, text='Quit Loading', command=lambda: image_btn_var.set(2))
        image_btn_stop.place(anchor=SE, relx=1, rely=1, relwidth=0.25, relheight=0.07)

        # calculate max resolution posible for image
        res = int(min(face_frame.winfo_height(), face_frame.winfo_width()) * 0.8)

        for image in images:
            faces = l.extract_face(DIR, image, confidence_factor.get())

            # after extract get each face weait for button and save
            for face in faces:
                load_img = ImageTk.PhotoImage(Image.fromarray(face).resize((res,res),1))
                image_label.config(image=load_img)
                image_btn.wait_variable(image_btn_var)
                if (image_btn_var.get() == 2):
                    break
                l.save_face(face, image_ety_var.get(), DIR)
            
            if (image_btn_var.get() == 2):
                break

        # when no images or quit loading
        image_label.destroy()
        image_btn.destroy()
        image_btn_stop.destroy()
        image_entry.destroy()
        load_btn.place(relx=0.5, rely=0.5, anchor=CENTER)


face_frame = Frame(load_frame, bg=W_BGCOL)
face_frame.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.9)

load_btn = Button(face_frame, text='Select Image Folder...', command=analize_images)
load_btn.place(relx=0.5, rely=0.5, anchor=CENTER)


####################### search layout #######################

# search for the images prints results and more
def search_coincidences(face):
    res = int(results_frame.winfo_width() * 0.95 /2)

    # set ups layout
    in_image_label = Label(results_frame)
    in_image_label.place(relx=0, rely=0, anchor=NW)
    original_img = ImageTk.PhotoImage(Image.fromarray(face).resize((res,res),1))
    in_image_label.config(image=original_img)

    out_image_label = Label(results_frame)
    out_image_label.place(relx=0, rely=0, anchor=NW)

    result_label = Label(results_frame, bg=W_BGCOL, anchor=CENTER, fg=W_FGCOL)

    image_btn_var = IntVar()
    image_btn = Button(results_frame, text='Quit Search', command=lambda: image_btn_var.set(1))
    image_btn.place(anchor=S, relx=0.5, rely=1)

    # search for coincidences
    (f, name, err) = l.search_image(face, DIR, algorithm.get() == 2)

    # prints results
    match_img = ImageTk.PhotoImage(Image.fromarray(f).resize((res,res),1))
    out_image_label.config(image=match_img)
    result_label.configure(text='Name: {} Error: {}'.format(name, err))
    result_label.place(anchor=S, relx=0.5, rely=0.85)


    image_btn.wait_variable(image_btn_var)
    
    # cleans layout
    in_image_label.destroy()
    out_image_label.destroy()
    result_label.destroy()
    image_btn.destroy()
    search_btn.place(relx=0.5, rely=0.5, anchor=CENTER)


# analizes a single image and selects only one face
def analize_single_image():
    # ask for image path
    image = Filedialog.askopenfilename( initialdir=os.getcwd(), title="Select a File", filetypes=(("All Files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")))
    if len(image) == 0:
        return
    
    faces = l.extract_face(DIR, image, confidence_factor.get())
    if (len(faces) == 0):
        return

    # set layout
    search_btn.place_forget()

    image_label = Label(results_frame)
    image_label.pack(side=TOP)

    image_btn_var = IntVar()
    image_btn = Button(results_frame, text='Next Face', command=lambda: image_btn_var.set(1))
    image_btn.place(anchor=SE, relx=0.9, rely=1, relwidth=0.375)

    image_btn_stop = Button(results_frame, text='Search This Face', command=lambda: image_btn_var.set(2))
    image_btn_stop.place(anchor=SW, relx=0.1, rely=1, relwidth=0.375)

    # calculate max resolution posible for image
    res = int(min(results_frame.winfo_height(), results_frame.winfo_width()) * 0.8)

    # after extract get each face weait for button and save
    for face in faces:
        search_img = ImageTk.PhotoImage(Image.fromarray(face).resize((res,res),1))
        image_label.config(image=search_img)
        final_face = face
        image_btn.wait_variable(image_btn_var)
        if (image_btn_var.get() == 2):
            break
    
    # unloads al widgets
    image_label.destroy()
    image_btn.destroy()
    image_btn_stop.destroy()

    # searches and shows results of the face
    search_coincidences(final_face)
    

results_frame = Frame(search_frame, bg=W_BGCOL)
results_frame.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.9)

search_btn = Button(search_frame, text='Select Image ...', command=analize_single_image)
search_btn.place(relx=0.5, rely=0.5, anchor=CENTER)

root.mainloop()