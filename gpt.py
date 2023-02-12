import tensorflow as tf
import tensorflow_addons as tfa


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim, expansion, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.expansion = dim, expansion
        # 2 dense layers with specified expansion
        self.dense1 = tf.keras.layers.Dense(self.dim * self.expansion, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(self.dim, activation=None)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        return {"dim": self.dim, "expansion": self.expansion, "dropout_rate": self.dropout_rate}

    def call(self, inputs):
        hidden = self.dense1(inputs)
        hidden = self.dense2(hidden)
        return self.dropout(hidden)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout_rate = dim, heads, dropout_rate
        # weigth matrices
        self.W_Q = self.add_weight(shape=[self.dim, self.dim], initializer="glorot_uniform",
                                   dtype=tf.float32, trainable=True, name="W_Q")
        self.W_K = self.add_weight(shape=[self.dim, self.dim], initializer="glorot_uniform",
                                   dtype=tf.float32, trainable=True, name="W_K")
        self.W_V = self.add_weight(shape=[self.dim, self.dim], initializer="glorot_uniform",
                                   dtype=tf.float32, trainable=True, name="W_V")
        self.W_O = self.add_weight(shape=[self.dim, self.dim], initializer="glorot_uniform",
                                   dtype=tf.float32, trainable=True, name="W_P")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        return {"dim": self.dim, "heads": self.heads, "dropout_rate": self.dropout_rate}

    def call(self, inputs):
        shape_inputs = tf.shape(inputs)

        Q = inputs @ self.W_Q  # queries
        K = inputs @ self.W_K  # keys
        V = inputs @ self.W_V  # values

        # [batch_size, time, heads, dim]
        Q = tf.reshape(Q, [shape_inputs[0], shape_inputs[1], self.heads, self.dim // self.heads])
        K = tf.reshape(K, [shape_inputs[0], shape_inputs[1], self.heads, self.dim // self.heads])
        V = tf.reshape(V, [shape_inputs[0], shape_inputs[1], self.heads, self.dim // self.heads])

        # [batch_size, heads, time, dim]
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        # self-attention weights
        sa_weights = tf.matmul(Q, K, transpose_b=True)
        sa_weights = sa_weights / tf.math.sqrt(tf.cast(self.dim // self.heads, tf.float32))

        # mask - our model won't look into the future
        mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones(tf.shape(sa_weights))).to_dense()
        mask = tf.cast(mask, tf.bool)

        sa_weights = tf.keras.layers.Softmax()(sa_weights, mask)
        sa_weights = self.dropout1(sa_weights)
        values = sa_weights @ V
        values = tf.transpose(values, perm=[0, 2, 1, 3])
        values = tf.reshape(values, [shape_inputs[0], shape_inputs[1], self.dim])
        return self.dropout2(values @ self.W_O)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim, self.heads, self.dropout_rate = dim, heads, dropout_rate

        self.sa = MultiHeadAttention(dim, heads, dropout_rate)
        self.ffwd = FFN(dim, 4, dropout_rate)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def get_config(self):
        return {"dim": self.dim, "heads": self.heads, "dropout_rate": dropout_rate}

    def call(self, inputs):
        # using residuals connections
        inputs = inputs + self.sa(self.ln1(inputs))
        inputs = inputs + self.ffwd(self.ln2(inputs))
        return inputs


class GPT(tf.keras.Model):
    def __init__(self, vocab_size, n_embd, block_size, n_layers, n_heads, dropout_rate):
        self.block_size = block_size
        inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        B, T = tf.shape(inputs)

        token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)(inputs)
        position_embedding_table = tf.keras.layers.Embedding(block_size, n_embd)(tf.range(T))

        x = token_embedding_table + position_embedding_table
        for _ in range(n_layers):
            x = TransformerBlock(n_embd, n_heads, dropout_rate)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        logits = tf.keras.layers.Dense(vocab_size)(x)

        super().__init__(inputs=inputs, outputs=logits)
        self.compile(optimizer=tfa.optimizers.AdamW(3e-4),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    def train(self, train_data: tf.Tensor, val_data: tf.Tensor = None, n_steps: int = 10_000,
              batch_size: int = 32, block_size: int = 8, eval_interval: int = 1_000):

        losses = tf.Variable(tf.zeros(n_steps))
        validation = val_data is not None
        if validation:
            losses_val = tf.Variable(tf.zeros(100))

        # training
        for steps in range(n_steps):
            # get batch function
            xb, yb = self.get_batch(train_data, batch_size, block_size)
            with tf.GradientTape() as tape:
                logits = self(xb, training=True)
                loss = self.loss(yb, logits)
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
            losses[steps].assign(loss)

            # evaluation
            if (steps + 1) % eval_interval == 0:
                if validation:
                    for eval_step in range(100):
                        xb_val, yb_val = self.get_batch(val_data, batch_size, block_size)
                        logits = self(xb_val, training=False)
                        losses_val[eval_step].assign(self.loss(yb_val, logits))
                    val_loss = tf.math.reduce_mean(losses_val)
                tf.print(
                    f"step {steps + 1}: train loss {tf.math.reduce_mean(losses[steps + 1 - eval_interval:steps + 1]):.4f}",
                    end="")
                if validation:
                    tf.print(f", validation loss {val_loss:.4f}", end="")
                tf.print()

    def get_batch(self, data: tf.Tensor, batch_size: int, block_size: int):
        ix = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(data) - block_size, dtype=tf.int32)
        xb = tf.stack([data[i:i + block_size] for i in ix])
        yb = tf.stack([data[i + 1:i + block_size + 1] for i in ix])
        return xb, yb

    def generate(self, idx, max_new_tokens):
        # generating text
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -256:]
            # get the predictions
            logits = self(idx_cond)
            # sampling next id - `tf.random.categorical` takes logits as argument
            id_next = tf.random.categorical(logits[:, -1, :], 1, dtype=tf.int32)
            # appending to the idx
            idx = tf.concat((idx, id_next), axis=-1)
        return idx

