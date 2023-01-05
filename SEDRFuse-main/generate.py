# Use a trained SEDRFuse Net to generate fused images
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from max_fuse import max_fuse
from datetime import datetime
# from fusion_l1norm import L1_norm
from fusion_attention import attention
from resnet_encoder_decoder import resnet_decoder,resnet_encoder
from utils import get_images, save_images, get_train_images, get_train_images_rgb


def generate(infrared_path, visible_path, model_path, index, IS_RGB, output_path=None):

	if IS_RGB:
		print('RGB image fusion')
		_handler_rgb_att(infrared_path, visible_path, model_path, index, output_path=output_path)
	else:
		print('Gray image fusion')
		_handler_att(infrared_path, visible_path, model_path, index, output_path=output_path)



def _handler_att(ir_path, vis_path, model_path, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)

	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	output = handler(ir_img, vis_img, model_path, index)

	save_images(ir_path, output, output_path, prefix='F' + str('%03d'%index))
		
def _handler_rgb_att(ir_path, vis_path, model_path, index, output_path=None):

	ir_img = get_train_images_rgb(ir_path, flag=False)/255
	vis_img = get_train_images_rgb(vis_path, flag=False)/255
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])


	ir_img1 = ir_img[:, :, :, 0]
	ir_img1 = ir_img1.reshape([1, dimension[0], dimension[1], 1])
	ir_img2 = ir_img[:, :, :, 1]
	ir_img2 = ir_img2.reshape([1, dimension[0], dimension[1], 1])
	ir_img3 = ir_img[:, :, :, 2]
	ir_img3 = ir_img3.reshape([1, dimension[0], dimension[1], 1])

	vis_img1 = vis_img[:, :, :, 0]
	vis_img1 = vis_img1.reshape([1, dimension[0], dimension[1], 1])
	vis_img2 = vis_img[:, :, :, 1]
	vis_img2 = vis_img2.reshape([1, dimension[0], dimension[1], 1])
	vis_img3 = vis_img[:, :, :, 2]
	vis_img3 = vis_img3.reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1.shape)

	output1 = handler(ir_img1, vis_img1, model_path, index)
	output2 = handler(ir_img2, vis_img2, model_path, index)
	output3 = handler(ir_img3, vis_img3, model_path, index)


	output1 = output1.reshape([1, dimension[0], dimension[1]])
	output2 = output2.reshape([1, dimension[0], dimension[1]])
	output3 = output3.reshape([1, dimension[0], dimension[1]])

	output = np.stack(( output3, output2, output1), axis=-1)

	save_images(ir_path, output, output_path, prefix='fused' + str('%03d'%index))

def handler(ir_img, vis_img, model_path, index):

	print('img shape final:', ir_img.shape)

	g1 = tf.Graph()
	g2 = tf.Graph()
	g3 = tf.Graph()

	with g1.as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')

		IR_C1_64, IR_C2_128, IR_Res_256 = resnet_encoder(infrared_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		IR_C1_temp, IR_C2_temp, IR_Res_temp = sess.run([IR_C1_64, IR_C2_128, IR_Res_256], feed_dict={infrared_field: ir_img})

	with g2.as_default(), tf.Session() as sess:

		visible_field = tf.placeholder(
			tf.float32, shape=vis_img.shape, name='style')

		VS_C1_64, VS_C2_128, VS_Res_256 = resnet_encoder(visible_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		VS_C1_temp, VS_C2_temp, VS_Res_temp = sess.run([VS_C1_64, VS_C2_128, VS_Res_256], feed_dict={visible_field: vis_img})

	with g3.as_default(), tf.Session() as sess:

		Fuse_C1 = max_fuse(IR_C1_temp, VS_C1_temp)
		Fuse_C2 = max_fuse(IR_C2_temp, VS_C2_temp)
		Fuse_Res = attention(IR_Res_temp, VS_Res_temp)

		Fuse_C1_field = tf.placeholder(
			tf.float32, shape=Fuse_C1.shape, name='COMP1')
		
		Fuse_C2_field = tf.placeholder(
		    tf.float32, shape=Fuse_C2.shape, name='COMP2')

		Fuse_Res_field = tf.placeholder(
		    tf.float32, shape=Fuse_Res.shape, name='INTER')

		Decode_feature = resnet_decoder(Fuse_C1_field, Fuse_C2_field, Fuse_Res_field)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output = sess.run(Decode_feature, feed_dict={Fuse_Res_field:Fuse_Res, Fuse_C2_field:Fuse_C2, Fuse_C1_field:Fuse_C1})

	return output