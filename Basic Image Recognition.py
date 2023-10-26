from PIL import Image
import numpy as np
from IPython.display import display

a = Image.open('/content/boy.png')
b = Image.open('/content/anadearmas.jpg')
i1 = np.array(a)
i2 = np.array(b)

def imageresize(im, sz):
    # Resizing an image using PIL
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))

a1 = Image.fromarray(i1)  # convert array1 to image
b1 = Image.fromarray(i2)  # convert array2 to image

display(a1)
display(b1)
