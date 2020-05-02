from modules import *
import tensorflow.compat.v1 as tf


class SASRec():
    def __init__(self, max_item, args, reuse=None):

        self.is_training = tf.placeholder(tf.bool, shape=())
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=None)
        pos = self.pos
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=max_item + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training),
                                         seed=args.random_seed)

            self.seq *= mask

            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   seed=args.random_seed,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training,
                                           seed=args.random_seed)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        # find representation
        self.rep = self.seq[:, -1, :]


        seq_emb = tf.reshape(self.rep, [tf.shape(self.input_seq)[0], args.hidden_units])
        indices = pos - 1
        labels = tf.one_hot(indices, max_item)
        item_emb = tf.nn.embedding_lookup(item_emb_table, tf.range(1, max_item + 1))
        logits = tf.matmul(seq_emb, tf.transpose(item_emb))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # prediction
        self.test_item = tf.placeholder(tf.int32, shape=None)
        self.test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(self.test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], tf.shape(self.test_item)[0]])
        self.pred_last = tf.argsort(tf.argsort(-self.test_logits))

    def predict(self, sess, seq, item_idx):
        return sess.run(self.pred_last,
                        {self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
