# TPE PCA - Face Recongnition Software based on PCA Analysis

The face recognition software is based on a series of steps (and a python script for each one) detailed in the next sections.

> Is required to have installed Python 3. On Ubuntu run `sudo apt update` and `sudo apt install python3-pip`.  
To check versions run `python3 --version` or `pip3 --version`.

---

### 1. Face extraction
Here the program gets the faces from the image, based on the ?
```bash
# install packages previously
pip3 install numpy
pip3 install opencv-python
```

```bash
# generic example how to run
python3 face_extract.py --image path/to/image.jpeg --prototxt path/to/file.prototxt --model path/to/file.caffemodel --confidence 0.2 --output path/to/save
```

```bash
# specific example, run inside mna-tpe1
python3 face_extract/face_extract.py --image image_samples/foto2.jpeg --prototxt face_extract/deploy.prototxt --model face_extract/res10_300x300_ssd_iter_140000.caffemodel --confidence 0.2 --output face_extract
```


