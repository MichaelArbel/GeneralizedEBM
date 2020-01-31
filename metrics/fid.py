# example of calculating the frechet inception distance in Keras
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras import backend as K
# scale an array of images to a new size





def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance

def compute_stats(model,images,b_size):
	images = scale_images(images , (299,299,3))
	images = preprocess_input(images)
	# m=0
	# for _ in range(n_batches):
	# 	if m<images.shape[0]:

	# 		lengthstuff= min(images.shape[0]-m,b_size)
	# 		act_b = model.predict(images[m:m+lengthstuff,:])
	# 		if m==0:
	# 			activations = K.placeholder(shape=(None,)+images.shape[1:] )
	# 		activations[m:m+lengthstuff,:]=act_b
	# 		m = m + act_b.size(0)
	act = model.predict(images)
	#act = model.predict(images)
	mu, sigma = act.mean(axis=0), cov(act, rowvar=False)
	return mu,sigma


def compute_fid(mu1, sigma1, mu2, sigma2):
	# calculate activations
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid