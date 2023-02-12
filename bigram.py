import tensorflow as tf
import tensorflow_addons as tfa


class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size):
        inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        token_embedding_table = tf.keras.layers.Embedding(vocab_size, vocab_size)(inputs)

        super().__init__(inputs=inputs, outputs=token_embedding_table)
        self.compile(optimizer=tfa.optimizers.AdamW(0.001),
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
                tf.print(f"step {steps+1}: train loss {tf.math.reduce_mean(losses[steps+1-eval_interval:steps+1]):.4f}", end="")
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
            # get the predictions
            logits = self(idx)
            # sampling next id - `tf.random.categorical` takes logits as argument
            id_next = tf.random.categorical(logits[:, -1, :], 1, dtype=tf.int32)
            # appending to the idx
            idx = tf.concat((idx, id_next), axis=-1)
        return idx

