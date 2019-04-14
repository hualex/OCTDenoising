import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mimg
from scipy import ndimage

def get_local_map(img, kernel_size =3):
    """
    return image local mean map and local standard deviation map.

    kernel_size: the size of image local 

    """
    img_local_mean_map = ndimage.uniform_filter(img,size=kernel_size,mode="reflect")
    img_sqr_map = ndimage.uniform_filter(img**2,size=kernel_size,mode="reflect")
    img_local_var_map = img_sqr_map-img_local_mean_map**2 
    img_local_std_map = np.sqrt(img_local_var_map)
    return img_local_mean_map,img_local_std_map



class Preprocessing(object):

    def __call__(self, sample):
        
        img = rgb2gray(sample)
        img_mean_map ,img_dev_map = get_local_map(img)
        return (img-img_mean_map)/img_dev_map

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def get_local_map(img, kernel_size =3):
    """
    return image local mean map and local standard deviation map.

    kernel_size: the size of image local 

    """
        img_local_mean_map = ndimage.uniform_filter(img,size=kernel_size,mode="reflect")
        img_sqr_map = ndimage.uniform_filter(img**2,size=kernel_size,mode="reflect")
        img_local_var_map = img_sqr_map-img_local_mean_map**2 
        img_local_std_map = np.sqrt(img_local_var_map)
        return img_local_mean_map,img_local_std_map





if __name__ == "__main__":

    img = rgb2gray(plt.imread('Lenna.png').astype(float))
    img_mean_map ,img_dev_map = get_local_map(img)
    img_mean_map2 ,img_dev_map2 = get_local_map(img,kernel_size=3)
    fig = plt.figure()

    fig.add_subplot(1, 3, 1, title='Original')
    plt.imshow((img-img_mean_map),cmap='gray')
    
    fig.add_subplot(1, 3, 2, title='Mean Map')
    plt.imshow(img_mean_map2,cmap = 'gray')
    
    fig.add_subplot(1, 3, 3, title='Deviation Map')
    plt.imshow(img_dev_map2,cmap = 'gray')
    plt.show()
