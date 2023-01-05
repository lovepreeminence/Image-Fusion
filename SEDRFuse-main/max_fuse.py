import numpy as np
import tensorflow as tf

def max_fuse(source_en_a, source_en_b):
    result = []
    narry_a = source_en_a
    narry_b = source_en_b
    dimension = source_en_a.shape

    for i in range(dimension[3]):

        featureA = []
        featureB = []

        featureA = narry_a[0,:,:,i]
        featureB = narry_b[0,:,:,i]

        temp_matrix = np.maximum(featureA,featureB)

        result.append(temp_matrix)

    result = np.stack(result, axis=-1)

    result_tf = np.reshape(result, (dimension[0], dimension[1], dimension[2], dimension[3]))

    return result_tf
