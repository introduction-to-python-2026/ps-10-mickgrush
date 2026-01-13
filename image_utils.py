from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def load_image(path):
  image = Image.open(path)
  image_arr = np.array(image)
  return image_arr

def edge_detection(image):
  #grayscale
  grayscale_image = np.mean(image, axis=2)
  #gray scale show: plt.imshow(grayscale_image, cmap = 'gray')


  # edge finding
  kernelx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  kernely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

  edge_x = convolve2d(grayscale_image, kernelx, mode='same', boundary='fill', fillvalue=0)
  edge_y = convolve2d(grayscale_image, kernely, mode='same', boundary='fill', fillvalue=0)
  #edgeMAG
  edgeMAG = (edge_x**2 + edge_y**2)**(1/2)

  return edgeMAG
