from tkinter import *
from tkinter import messagebox, filedialog, simpledialog
#from db import Database
from PIL import Image, ImageTk
import glob
import tkinter as tk

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
        
def open_image():
    global my_image
    image_path = filedialog.askopenfilename( initialdir="/Users/geronimomaspero/Desktop/part_manager", title="Select a File", filetypes=( ("All Files", "*.*"), ("png files", "*.png"), ("jpg files", "*.jpg")))
    my_image = ImageTk.PhotoImage(Image.open(image_path))
    #creo un label y a ese label le cargo la imagen
    my_image_label = Label(image=my_image)
    #Ubico a mi label en pantalla
    my_image_label.place(height=400, width=400,x=400,y=200)

    # Display Label 0
    label5 = Label(app, text="Potential name: John Cena", font=('bold', 14))
    label5.place(height=20, width=400,x=400,y=600)

    # Display Label 1
    label6 = Label(app, text="Percentage of similarity = 98.4%", font=('bold', 14))
    label6.place(height=20, width=400,x=400,y=630)

    # Display Label 2
    label7 = Label(app, text="Dick size = 20 inches", font=('bold', 14))
    label7.place(height=20, width=400,x=400,y=660)
    

        
def popupmessage(path_variable):
    #Build windows
    popup = tk.Toplevel(app)
    popup.wm_title("Name Assignment")
    popup.geometry('500x500')

    # Display Label
    label = Label(popup, text="Please assign the correct name to the following face", font=('bold', 14))
    label.grid(row=0, column=2)

    # Obtain image from the path_variable (argument) 
    global my_image2
    my_image2 = ImageTk.PhotoImage(Image.open(path_variable))
    
    # Display image in the popup
    my_image_label2 = Label(popup,image=my_image2)
    my_image_label2.grid(row=1, column=2)

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
    path = filedialog.askdirectory( initialdir="/Users/geronimomaspero/Desktop/part_manager", title="Select a File")
    #this is a list with all the images' paths.
    list_of_items = glob.glob(path + '/*.png')
    for file in list_of_items:
        #Popeamos una ventanita que le pida nombre para la imagen y la meta en la scroll list.
        name = popupmessage(file)
        a = name.get()
        # Ya tengo la imagen y su respectivo nombre

        


    





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