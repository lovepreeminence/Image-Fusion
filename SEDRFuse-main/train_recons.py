# Train the SEDRFuse Network
from __future__ import print_function

import numpy as np
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils import get_train_images
from ssim_loss_function import SSIM_LOSS
from resnet_encoder_decoder import resnet_decoder,resnet_encoder

# TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
# TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE = (1024, 1280, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (1024, 1280, 1) # (height, width, color_channels)

LEARNING_RATE = 1e-4
EPSILON = 1e-5
ssim_weight = 1

def train_recons(original_imgs_path, validatioin_imgs_path, save_path, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1000):
    if debug:
        start_time = datetime.now()
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)

    num_val = len(validatioin_imgs_path)
    num_imgs = len(original_imgs_path)

    # num_imgs = 50

    original_imgs_path = original_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    print('Train images number    : %d.\n' % num_imgs)
    print('Train images samples   : %s.\n' % str(num_imgs / BATCH_SIZE))
    print('Validate images number : %d.\n' % num_val)

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        source = original

        print('source  :', source.shape)
        print('original:', original.shape)

        ###############################create a symmetric encoder-decoder with resnet #######################

        feature_c1,feature_c2,feature_res = resnet_encoder(source,gf_dim=64)
        generated_img = resnet_decoder(feature_c1, feature_c2, feature_res, gf_dim=64, output_c_dim=1)

        #####################################################################################################

        print('generate:', generated_img.shape)

        ssim_loss_value = SSIM_LOSS(original, generated_img)
        pixel_loss = tf.reduce_sum(tf.square(original - generated_img))
        pixel_loss = pixel_loss/(BATCH_SIZE*HEIGHT*WIDTH)
        ssim_loss = 1 - ssim_loss_value

        loss = ssim_weight*ssim_loss + pixel_loss
       
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,max_to_keep=300)

        ###########################  **Start Training**  ##################################################
        step = 0
        count_loss = 0
        n_batches = int(len(original_imgs_path) // BATCH_SIZE)
        val_batches = int(len(validatioin_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')
            start_time = datetime.now()

        Loss_all = [i for i in range(EPOCHS * n_batches)]
        Loss_ssim = [i for i in range(EPOCHS * n_batches)]
        Loss_pixel = [i for i in range(EPOCHS * n_batches)]
        Val_ssim_data = [i for i in range(EPOCHS * n_batches)]
        Val_pixel_data = [i for i in range(EPOCHS * n_batches)]
        for epoch in range(EPOCHS):

            np.random.shuffle(original_imgs_path)

            for batch in range(n_batches):
                # retrive a batch of content and style images

                original_path = original_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                # original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                original_batch = original_batch.reshape([BATCH_SIZE, 1024, 1280, 1])

                # print('original_batch shape final:', original_batch.shape)

                # run the training step
                sess.run(train_op, feed_dict={original: original_batch})
                step += 1
              
                elapsed_time = datetime.now() - start_time

                print('epoch: %d/%d, step: %d/%d, elapsed time: %s' % (epoch, EPOCHS, step, n_batches, elapsed_time))

                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                if is_last_step or step % logging_period == 0:
                    elapsed_time = datetime.now() - start_time

                    _ssim_loss, _loss, _p_loss = sess.run([ssim_loss, loss, pixel_loss], feed_dict={original: original_batch})
                    Loss_all[count_loss] = _loss
                    Loss_ssim[count_loss] = _ssim_loss
                    Loss_pixel[count_loss] = _p_loss

                    print('--------------------------------------------------------------------------------------------------------------------------------')
                    print('Training Value:')
                    print('epoch: %d/%d, step: %d' % (epoch, EPOCHS, step))
                    print('p_loss: %s, ssim_loss: %s,  total loss: %s, elapsed time: %s' % (_p_loss, _ssim_loss, _loss, elapsed_time))
                    print('--------------------------------------------------------------------------------------------------------------------------------')

                    # calculate the accuracy rate for 1000 images, every 1000 steps
                    val_ssim_acc = 0
                    val_pixel_acc = 0
                    np.random.shuffle(validatioin_imgs_path)
                    val_start_time = datetime.now()
                    for v in range(val_batches):
                        val_original_path = validatioin_imgs_path[v * BATCH_SIZE:(v * BATCH_SIZE + BATCH_SIZE)]
                        val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT, crop_width=WIDTH,flag=False)
                        # val_original_batch = val_original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                        val_original_batch = val_original_batch.reshape([BATCH_SIZE, 1024, 1280, 1])
                        val_ssim, val_pixel = sess.run([ssim_loss, pixel_loss], feed_dict={original: val_original_batch})
                        val_ssim_acc = val_ssim_acc + (1 - val_ssim)
                        val_pixel_acc = val_pixel_acc + val_pixel
                    Val_ssim_data[count_loss] = val_ssim_acc/val_batches
                    Val_pixel_data[count_loss] = val_pixel_acc / val_batches
                    val_es_time = datetime.now() - val_start_time
                    print('--------------------------------------------------------------------------------------------------------------------------------')
                    print('Validation Value:')
                    print('Pixel: %s, SSIM: %s, Total: %s, elapsed time: %s' % (val_pixel_acc / val_batches, val_ssim_acc/val_batches, (val_pixel_acc + val_ssim_acc) / val_batches, val_es_time))
                    print('--------------------------------------------------------------------------------------------------------------------------------')
                    count_loss += 1
            
            # if you want to find the best performance, you can save the model at each epoch 
            saver.save(sess, save_path[0].format(epoch))

        # ** Done Training & Save the last model **
        saver.save(sess, save_path[1])

        loss_data = Loss_all[:count_loss]
        scio.savemat('./models/loss/SEDRLossData'+'.mat',{'loss':loss_data})

        loss_ssim_data = Loss_ssim[:count_loss]
        scio.savemat('./models/loss/SEDRLossSSIMData'+'.mat', {'loss_ssim': loss_ssim_data})

        loss_pixel_data = Loss_pixel[:count_loss]
        scio.savemat('./models/loss/SEDRLossPixelData'+'.mat', {'loss_pixel': loss_pixel_data})

        # validation_ssim_data = Val_ssim_data[:count_loss]
        # scio.savemat('./models/loss/Validation_ssim_Data'  + '.mat', {'val_ssim': validation_ssim_data})

        # validation_pixel_data = Val_pixel_data[:count_loss]
        # scio.savemat('./models/loss/Validation_pixel_Data' + '.mat', {'val_pixel': validation_pixel_data})


        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Last model is saved to: %s' % save_path[1])

