import tensorflow as tf
import tensorflow.keras.backend as K


class Hyper_Model(tf.keras.Model):
    def __init__(self, embedding_matrix=None, hinge_margin=0.0):
        super(HYPER, self).__init__()
        self.hinge_margin = margin
        if embedding_matrix:
            self.embedding = tf.keras.Embedding(300, embeddings_initializer=tf.keras.initializers.Constant(values=embedding_matrix))
        else:
            self.embedding = tf.keras.Embedding(300, embeddings_initializer="uniform")
        self.projection = tf.keras.layers.Dense(300, activation="relu")
        self.representation = tf.keras.layers.Lambda(self.bow_representation)
        self.distance = tf.keras.layers.Lambda(self.distance)
        self.wf = tf.Variable(tf.keras.initializer(sh))

    
    def bow_representation(self, embedding_sequence):
        embedding_sum = tf.squeeze(tf.math.reduce_sum(embedding_sequence, axis=2))
        normalized_sum = embedding_sum / tf.norm(embedding_sum, axis=-1, keep_dims=True)
        return normalized_sum
                
    def distance(self, q, a):
        num = tf.square(tf.norm(q-a, axis=1))
        den = (1 - tf.square(tf.norm(q, axis=1))) * (1 - tf.square(tf.norm(a, axis=1)))
        distance = tf.math.acosh(1 + 2 * num / den)
        return distance
    
    

    
        