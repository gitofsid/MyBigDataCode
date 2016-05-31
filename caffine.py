"""
/* ******************************************************************************************
 * Name - Siddharth Baronia 
 *
 * Problems - 1. Use caffenet to use image prediction and display label
 *            2. Visualize different intermediate layers
 * ********************************************************************************************/ 
"""

# set up Python environment: numpy for numerical routines, and matplotlib for plotting

import pylab
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# display plots in this notebook


def vis_square(data,image,conv):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
    title = image + ' ' + conv
    plt.title(title,fontweight='bold')
    pylab.imshow(data)
    plt.pause(2)
    plt.close()


# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = ''  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system("scripts/download_model_binary.py models/bvlc_reference_caffenet")

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


output_to = sys.argv[1]
output_file = open(output_to, "w+")

with open("val.txt", "r") as lines:
	count = 0
	for line in lines:
		if count == 100:
			print "Validated 100 images. Exiting"
			output_file.close()
			exit(1)

		image = caffe.io.load_image(caffe_root + 'VOC2012/JPEGImages/' + line.strip("\n\r") + '.jpg')
		transformed_image = transformer.preprocess('data', image)
		plt.imshow(image)
		pylab.imshow(image)
		plt.pause(2)


		# copy the image data into the memory allocated for the net
		net.blobs['data'].data[...] = transformed_image

		### perform classification
		output = net.forward()

		output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

		print 'predicted class is:', output_prob.argmax()


		# load ImageNet labels
		labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
		if not os.path.exists(labels_file):
			os.system("data/ilsvrc12/get_ilsvrc_aux.sh")
		    
		labels = np.loadtxt(labels_file, str, delimiter='\t')

		print 'val image: ', line.strip("\n\r"), ' output label:', labels[output_prob.argmax()]

		output_file.write('val image: ' + line.strip("\n\r") + ' predicted class is: ' + str(output_prob.argmax()) + ' output label: ' + str(labels[output_prob.argmax()]) + '\n')



		# sort top five predictions from softmax output
		top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

		print 'probabilities and labels:' 
		print zip(output_prob[top_inds], labels[top_inds])

		# for each layer, show the output shape
		print "net.blobs\n"
		for layer_name, blob in net.blobs.iteritems():
			print layer_name + '\t' + str(blob.data.shape)

		for i in range(5):
			feat = net.blobs['conv' + str(i+1)].data[0, :96]
			vis_square(feat,line.strip("\n\r"),'conv' + str(i+1))

		count = count + 1








