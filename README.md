# TPE PCA - Face Recongnition Software based on PCA Analysis

The face recognition software is based on a series of steps (and a python script for each one) detailed in the next sections.

> It is required to have installed Python 3. On Ubuntu run `sudo apt update` and `sudo apt install python3-pip`.  
To check versions run `python3 --version` or `pip3 --version`.

---


### 1. Set the environmental variables
in the application.py file, change these variables to your respective paths:

```bash
DATABASE_PATH = "/Users/geronimomaspero/Desktop/mna-tpe1/data"
INITIAL_DIR = "/Users/geronimomaspero/Desktop/mna-tpe1"
```

### 2. Run the program
First, you will need to download Tkinter: https://www.activestate.com/products/tcl/downloads/ 

```bash
# execute the following commands
pip3 install Pillow 
pip3 install numpy
pip3 install opencv-python
python3 application.py
```

<!-- ### 2. Face extraction
Here the program gets the faces from the image, based on the ?
```bash
# install packages previously
pip3 install numpy
pip3 install opencv-python
```

```bash
# loads to the database the faces on the image
# data is the folder, path to the "database"
# image is the path to the image to check
# confidence value for face detection, optional
python3 load.py --path /path/to/data --image path/to/image.jpeg --confidence 0.2
```


### 3. Calculate values
```bash
# makes the required calculation of the already processed data
# data is the folder, path to the "database"
python3 calculate.py --path /path/to/data
```

### 4. Search for matches
```bash
# given a image checks for coincidences
# data is the folder, path to the "database"
# image is the path to the image to check
# confidence value for face detection, optional
python3 search.py --path /path/to/data --image /path/to/image.jpeg --confidence val
``` -->

