import tensorflow as tf

def riemannian_gradient(grad, eps=1E-5):
    ## based on https://github.com/vanzytay/WSDM2018_HyperQA/blob/master/model/utilities.py
    shape = grad.shape
    if (len(shape) >= 3):
        grad_norm = 1 - tf.square(tf.norm(grad, axis=[-2, -1], ord='euclidean', keep_dims=True))
    elif (len(shape) == 2):
        grad_norm = 1 - tf.square(tf.norm(grad, ord='euclidean', keep_dims=True))
    else:
        return grad
    
    scale_factor = (tf.square(grad_norm) + eps) / 4.0
    grad = grad * scale_factor
    return grad

def get_loss(inputs, margin=1.0):
    sim_pos, sim_neg = inputs
    loss = tf.reduce_mean(tf.maximum(0.0, margin + sim_neg - sim_pos))
    return loss