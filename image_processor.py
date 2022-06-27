import cv2
from PIL import Image
from numpy import asarray

RESIZE_IMAGE_SIZE = (150, 150)

# the function resizes an image as a new image.
# filename -> the file name of the image we want to resize.
# newfilename -> the new file name for the resized image.
def resize_image(filename, newfilename):
    image = cv2.imread(filename)
    image = cv2.resize(image, RESIZE_IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(newfilename, image)