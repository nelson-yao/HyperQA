import tensorflow as tf


class HyperQA(tf.keras.Model):
    def __init__(self, vocab_size, max_length, dropout=0.5, embedding_matrix=None):
        super().__init__()
        self.embedding_size = 300
        self.dropout = dropout
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer=tf.keras.initializers.Constant(value=embedding_matrix))
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer="uniform")

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.single_projection = tf.keras.layers.Dense(300, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal)
        self.sequence_projection = tf.keras.layers.TimeDistributed(self.single_projection, input_shape=(max_length, self.embedding_size))
        self.representation = tf.keras.layers.Lambda(self.bow_representation)
        self.distance_layer = tf.keras.layers.Lambda(self.hyperbolic_distance)
        self.wf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32, name="similarity_weights")
        self.bf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32, name="similarity_biases")

    def call(self, inputs, training=False, mask=None):
        if training:
            reference, good, bad = inputs
            return self.similarity_score(reference, good), self.similarity_score(reference, bad)
        else:
            reference, good = inputs
            return self.similarity_score(reference, good)

    def similarity_score(self, input1, input2):
        embedding_q1 = self.embedding(input1)
        embedding_q2 = self.embedding(input2)
        projection_q1 = self.sequence_projection(embedding_q1)
        projection_q2 = self.sequence_projection(embedding_q2)

        bow_q1 = self.bow_representation(projection_q1)
        bow_q2 = self.bow_representation(projection_q2)

        bow_q1 = self.dropout_layer(bow_q1)
        bow_q2 = self.dropout_layer(bow_q2)
        distance = self.hyperbolic_distance((bow_q1, bow_q2))
        similarity = self.wf * distance + self.bf
        return similarity

    def bow_representation(self, embedding_sequence):
        embedding_sum = tf.squeeze(tf.reduce_sum(embedding_sequence, axis=2))
        normalized_num = tf.clip_by_norm(embedding_sum, 1.0, 1, name="normalization")
        return normalized_num

    def hyperbolic_distance(self, inputs):
        input1, input2 = inputs
        num = tf.square(tf.norm(input1 - input2, axis=-1))
        den = (1 - tf.square(tf.norm(input1, axis=-1))) * (1 - tf.square(tf.norm(input2, axis=-1)))
        distance = tf.math.acosh(1 + 2 * num / den, name="inverse_hyperbolic_cosine")
        return distance