import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime

def attention(source_en_a, source_en_b):

    result = []
    narry_a = source_en_a
    narry_b = source_en_b

    # img1 = (narry_a.eval())
    # img2 = (narry_b.eval())
    # imge1 = np.asmatrix(img1)
    # imge2 = np.asmatrix(img2)

    # att1 = Image.fromarray(imge1 * 255.0 / imge1.max())
    # att2 = Image.fromarray(imge2 * 255.0 / imge2.max())
    # att1.show()
    # att2.show()

    dimension = source_en_a.shape

    # caculate L1-norm
    temp_abs_a = tf.abs(narry_a)
    temp_abs_b = tf.abs(narry_b)

    _l1_a = tf.nn.softmax(temp_abs_a,3)
    _l1_b = tf.nn.softmax(temp_abs_b,3)

    # create attention map

    att_a = temp_abs_a*_l1_a
    att_b = temp_abs_b*_l1_b
    
    _l1_a = tf.reduce_sum(att_a, 3)

    _l1_a = tf.reduce_sum(_l1_a, 0)

    _l1_b = tf.reduce_sum(att_b, 3)
    _l1_b = tf.reduce_sum(_l1_b, 0)

    l1_a = _l1_a.eval() # attention map of source image A
    l1_b = _l1_b.eval() # attention map of source image B

    # imge1 = np.asmatrix(l1_a)
    # imge2 = np.asmatrix(l1_b)

    # att1 = Image.fromarray(imge1 * 255.0 / imge1.max())
    # att2 = Image.fromarray(imge2 * 255.0 / imge2.max())
    # att1.show()
    # att2.show()


    for i in range(dimension[3]):

        mask_value = l1_a + l1_b

        mask_sign_a = l1_a/mask_value
        mask_sign_b = l1_b/mask_value
        
        array_MASK_a = mask_sign_a
        array_MASK_b = mask_sign_b

        temp_matrix = array_MASK_a*narry_a[0,:,:,i] + array_MASK_b*narry_b[0,:,:,i]
      
        result.append(temp_matrix)

    result = np.stack(result, axis=-1)

    result_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return result_tf
