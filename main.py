from image_utils import load_image, edge_detection
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image


image = load_image('/content/GettyImages-681946576-1024x716.jpg')
clean_image = median(image, ball(3))
edge_mag = edge_detection(clean_image)
threshold = np.percentile(edge_mag, 90)
edge_binary = edge_mag > threshold
edge_image = Image.fromarray(edge_binary)
edge_image.save('my_edges.png')
