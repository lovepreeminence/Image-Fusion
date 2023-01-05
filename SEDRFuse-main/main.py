# Demo - train the Symmetric Encoder-Decoder Network & use it to generate an image

from __future__ import print_function
import os
import time
import tensorflow as tf
from train_recons import train_recons
from generate import generate
from utils import list_images

os.environ["CUDA_VISIBLE_DEVICES"] = "0" ## use GPU or not

## True for training phase
# IS_TRAINING = True
IS_TRAINING = False

# True for RGB images fusion phase
is_RGB = True
# is_RGB = False

BATCH_SIZE = 2
EPOCHES = 10  ## default= 50


model_save_path = [
	'./models/model{}.ckpt',
	'./models/model.ckpt'] # for training


# MODEL_SAVE_PATHS = './models/mymodel.ckpt' # for test
MODEL_SAVE_PATHS = './models/mymodel.ckpt' # for test

output_save_path = './results/'

## test infrared and visible image path
# path = 'testImgs/gray_images/Crop_images/'
# path = 'testImgs/color_images/'
path = 'testImgs/LLVIP_test/'


def main():

	if IS_TRAINING:
		# training and validation data
		original_imgs_path = list_images('./data/train')
		validatioin_imgs_path = list_images('./data/test')

		print('\nBegin to train the network ...\n')
		train_recons(original_imgs_path, validatioin_imgs_path, model_save_path, EPOCHES, BATCH_SIZE, debug=True)
		print('\nSuccessfully! Done training...\n')

	else:
		model_path = MODEL_SAVE_PATHS
		print('\nBegin to generate pictures ...\n')

		for i in range(14):
			index = i + 1

			# gray images
			infrared = path + 'IR' + str(index) + '.jpg'
			visible = path + 'VIS' + str(index) + '.jpg'

			# RGB images
			# infrared = path + 'lytro-3-A.jpg'
			# visible = path + 'lytro-3-B.jpg'
			
			generate(infrared, visible, model_path, index, is_RGB, output_path = output_save_path)

start_t = time.time()
if __name__ == '__main__':
    	main()
end_t = time.time()
print('Total running time: %s' % (end_t-start_t))
