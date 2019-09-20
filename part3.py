import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('pink_lake.png')
plt.imshow(img)
# plt.show()

img_add = np.clip(img + 0.25, 0, 1)
plt.imsave('img_add.png', img_add)
plt.imshow(img_add)
# plt.show()

shape = img.shape

img_chan_0 = np.zeros((shape[0], shape[1]))
img_chan_0 = img[:shape[0], :shape[1], 0]
plt.imsave('img_chan_0.png', img_chan_0)

img_chan_1 = np.zeros((shape[0], shape[1]))
img_chan_1 = img[:shape[0], :shape[1], 1]
plt.imsave('img_chan_1.png', img_chan_1)

img_chan_2 = np.zeros((shape[0], shape[1]))
img_chan_2 = img[:shape[0], :shape[1], 2]
plt.imsave('img_chan_2.png', img_chan_2)

img_gray = np.zeros((shape[0],shape[1]))
img_gray = 0.299 * img[:shape[0], :shape[1], 0] + 0.587 * img[:shape[0], :shape[1], 1]\
           + 0.114 * img[:shape[0], :shape[1], 2]
plt.imsave('img_gray.png', img_gray)

img_crop = np.zeros((shape[0]//2, shape[1], shape[2]))
img_crop = img[:shape[0]//2, 0:shape[1], 0:shape[2]]
plt.imsave('img_crop.png', img_crop)

img_flip_vert = np.zeros(shape)
img_flip_vert = img[[shape[0]-1-i for i in range(shape[0])], :shape[1], :shape[2]]
plt.imsave('img_flip_vert.png', img_flip_vert)

