import numpy as np

def process_images(images):
    # assumes values between -1 and 1
    images = images*127.5+127.5
    if (images.max() > 255.) or (images.min() < .0):
        print('WARNING! Inception min/max violated: min = %f, max = %f. Clipping values.' % (images.min(), images.max()))
        images = np.clip(images,a_min = 0., a_max = 255.)