import tensorflow as tf


class HyperQA(tf.keras.Model):
    def __init__(self, vocab_size, max_length, projection_dim=300, dropout=0.5, embedding_matrix=None, embedding_size=300):
        super().__init__()
        self.embedding_size = embedding_size
        self.projection_dim = projection_dim
        self.dropout = dropout
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer=tf.keras.initializers.Constant(value=embedding_matrix), mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, output_dim=self.embedding_size, embeddings_initializer="uniform", mask_zero=True)

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.single_projection = tf.keras.layers.Dense(self.projection_dim, activation="relu", kernel_initializer=tf.keras.initializers.glorot_normal)
        self.sequence_projection = tf.keras.layers.TimeDistributed(self.single_projection, input_shape=(max_length, self.embedding_size))
        self.wf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32, name="similarity_weights")
        self.bf = tf.Variable(initial_value=tf.initializers.random_normal()((1,)), dtype=tf.float32, name="similarity_biases")
        
    def call(self, inputs, training=False, mask=None):
        if training:
            q1, q2, q3 = inputs
        else:
            q1, q2 = inputs

        self.embedding_q1 = self.embedding(q1)
        self.embedding_q2 = self.embedding(q2)

        self.projection_q1 = self.sequence_projection(self.embedding_q1)
        self.projection_q2 = self.sequence_projection(self.embedding_q2)

        self.bow_q1 = self.bow_representation(self.projection_q1)
        self.bow_q2 = self.bow_representation(self.projection_q2)

        self.bow_q1 = self.dropout_layer(self.bow_q1)
        self.bow_q2 = self.dropout_layer(self.bow_q2)

        self.distance_pos = self.hyperbolic_distance(self.bow_q1, self.bow_q2)
        self.similarity_pos = self.wf * self.distance_pos + self.bf


        if training:
            self.embedding_q3 = self.embedding(q3)
            self.projection_q3 = self.sequence_projection(self.embedding_q3)
            self.bow_q3 = self.bow_representation(self.projection_q3)
            self.bow_q3 = self.dropout_layer(self.bow_q3)
            self.distance_neg = self.hyperbolic_distance(self.bow_q1, self.bow_q3)
            self.similarity_neg = self.wf * self.distance_neg + self.bf
            return self.similarity_pos, self.similarity_neg

        else:
            return self.similarity_pos


    def bow_representation(self, embedding_sequence):
        embedding_sum = tf.reduce_sum(embedding_sequence, axis=2)
        normalized_num = tf.clip_by_norm(embedding_sum, 1.0, axes=1, name="normalization")
        return normalized_num

    def hyperbolic_distance(self, input1, input2, epsilon=1e-6):
        num = tf.square(tf.norm(input1 - input2, axis=1))
        den = (1.0 - tf.square(tf.norm(input1, axis=-1))) * (1.0 - tf.square(tf.norm(input2, axis=-1)))
        distance = tf.math.acosh(1 + 2 * num / (den + epsilon), name="inverse_hyperbolic_cosine")
        return distance
    