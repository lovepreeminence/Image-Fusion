import tensorflow as tf
from ops import *
from utils_resnet import *

def resnet_encoder(image, gf_dim=64, reuse=False, name="encoder"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x
            # return tf.concat([y,x],3) 

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        # c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c0 = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, gf_dim, 3, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, gf_dim*4, name='g_r1')
        # r2 = residule_block(r1, gf_dim*4, name='g_r2')
        # r3 = residule_block(r2, gf_dim*4, name='g_r3')
        # r4 = residule_block(r3, gf_dim*4, name='g_r4')
        # r5 = residule_block(r4, gf_dim*4, name='g_r5')
        # r6 = residule_block(r5, gf_dim*4, name='g_r6')
        # r7 = residule_block(r6, gf_dim*4, name='g_r7')
        # r8 = residule_block(r7, gf_dim*4, name='g_r8')
        # r9 = residule_block(r8, gf_dim*4, name='g_r9')
        return c1, c2, r1


def resnet_decoder(feature_c1, feature_c2, feature_res, gf_dim=64,output_c_dim=1, reuse=False, name="decoder"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        # skip connections from the encoder to preserve the details of the source images
        d1 = deconv2d(feature_res, gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d1 = tf.nn.relu(feature_c2 + d1)

        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.nn.relu(feature_c1 + d2)

        d2 = tf.pad(d2, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        prediction = tf.nn.relu(conv2d(d2, output_c_dim, 3, 1, padding='VALID', name='g_pred_c'))
        
        return prediction