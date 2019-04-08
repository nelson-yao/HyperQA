import tensorflow as tf


class HyperQA(tf.keras.Model):
    def __init__(self, vocab_size, max_length, embedding_matrix=None):
        super().__init__()
        self.embedding_size = 300
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer=tf.keras.initializers.Constant(value=embedding_matrix))
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer="uniform")
        self.single_projection = tf.keras.layers.Dense(300, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal)
        self.sequence_projection = tf.keras.layers.TimeDistributed(self.single_projection, input_shape=(max_length, self.embedding_size))
        self.representation = tf.keras.layers.Lambda(self.bow_representation)
        self.distance = tf.keras.layers.Lambda(self.distance)
        self.wf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32)
        self.bf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32)

    def call(self, inputs, training=False, mask=None):
        if training:
            reference, good, bad = inputs # each is batch * sequence length * embedding_size
        else:
            reference, good = inputs

        embedding_ref = self.embedding(reference)
        embedding_good = self.embedding(good)
        projection_reference = self.sequence_projection(embedding_ref)
        projection_good = self.sequence_projection(embedding_good)

        sum_reference = tf.reduce_sum(projection_reference,axis=2)
        sum_good = self.bow_representation(projection_good)
        distance_positive = self.distance((sum_reference, sum_good))
        similarity_positive = self.wf * distance_positive + self.bf

        if training:
            embedding_bad = self.embedding(bad)
            projection_bad = self.sequence_projection(embedding_bad)
            sum_bad = self.bow_representation(projection_bad)
            distance_negative = self.distance((sum_reference, sum_bad))
            similarity_negative = self.wf * distance_negative + self.bf
            return similarity_positive, similarity_negative
        else:
            return similarity_positive

    def bow_representation(self, embedding_sequence):
        embedding_sum = tf.squeeze(tf.reduce_sum(embedding_sequence, axis=2))
        normalized_sum = embedding_sum / tf.norm(embedding_sum, axis=-1, keep_dims=True)
        return normalized_sum
                
    def distance(self, inputs):
        q, a = inputs
        num = tf.square(tf.norm(q-a, axis=1))
        den = (1 - tf.square(tf.norm(q, axis=1))) * (1 - tf.square(tf.norm(a, axis=1)))
        distance = tf.math.acosh(1 + 2 * num / den)
        return distance
