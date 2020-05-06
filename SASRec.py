from modules import *
import tensorflow.compat.v1 as tf


class SASRec():
    def __init__(self, max_item, args, reuse=None):

        self.is_training = tf.placeholder(tf.bool, shape=())
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=None)
        self.lr = tf.placeholder(tf.float32, shape=())
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

        # EWC loss

        # find representation
        self.rep = self.seq[:, -1, :]
        seq_emb = tf.reshape(self.rep, [tf.shape(self.input_seq)[0], args.hidden_units])
        indices = pos - 1
        labels = tf.one_hot(indices, max_item)
        item_emb = tf.nn.embedding_lookup(item_emb_table, tf.range(1, max_item + 1))
        logits = tf.matmul(seq_emb, tf.transpose(item_emb))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # prediction
        self.test_item = tf.placeholder(tf.int32, shape=None)
        self.test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(self.test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], tf.shape(self.test_item)[0]])
        self.pred_last = tf.argsort(tf.argsort(-self.test_logits))

    # def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
    #     # computer Fisher information for each parameter
    #
    #     # initialize Fisher information for most recent task
    #     self.F_accum = []
    #     for v in range(len(self.var_list)):
    #         self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
    #
    #     # sampling a random class from softmax
    #     probs = tf.nn.softmax(self.y)
    #     class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
    #
    #     for i in range(num_samples):
    #         # select random input image
    #         im_ind = np.random.randint(imgset.shape[0])
    #         # compute first-order derivatives
    #         ders = sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list),
    #                         feed_dict={self.x: imgset[im_ind:im_ind + 1]})
    #         # square the derivatives and add to total
    #         for v in range(len(self.F_accum)):
    #             self.F_accum[v] += np.square(ders[v])
    #
    #     # divide totals by number of samples
    #     for v in range(len(self.F_accum)):
    #         self.F_accum[v] /= num_samples


    def predict(self, sess, seq, item_idx):
        return sess.run(self.pred_last,
                        {self.input_seq: seq, self.test_item: item_idx, self.is_training: False})


