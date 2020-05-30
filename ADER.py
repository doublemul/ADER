#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : SemesterProject
# @Author       : Xiaouy LIN
# @File         : ADER.py
# @Description  : ADER model file.
# The implemention of self-attentive recommender is modified based on https://github.com/kang205/SASRec
from modules import *
from util import *
import tensorflow.compat.v1 as tf
import tqdm


class ADER():
    def __init__(self, item_num, args, reuse=None):
        self.args = args
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=None)
        self.exemplar_logits = tf.placeholder(tf.float32, shape=(None, None))
        self.exemplar_pos = tf.placeholder(tf.int32, shape=None)
        self.max_item = tf.placeholder(tf.int32, shape=())
        self.max_item_pre = tf.placeholder(tf.int32, shape=())
        self.lr = tf.placeholder(tf.float32, shape=())
        self.dropout_rate = tf.placeholder(tf.float32, shape=())
        pos = self.pos
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=item_num + 1,
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
                                         rate=self.dropout_rate,
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
                                                   dropout_rate=self.dropout_rate,
                                                   seed=args.random_seed,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=self.dropout_rate, is_training=self.is_training,
                                           seed=args.random_seed)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        # find representation
        self.rep = self.seq[:, -1, :]

        self.variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        # del self.variables[1]

        # define loss
        seq_emb = tf.reshape(self.rep, [tf.shape(self.input_seq)[0], args.hidden_units])
        indices = pos - 1
        self.labels = tf.one_hot(indices, self.max_item)
        item_emb = tf.nn.embedding_lookup(item_emb_table, tf.range(1, self.max_item + 1))
        self.logits = tf.matmul(seq_emb, tf.transpose(item_emb))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.gradient = tf.gradients(self.loss, self.variables)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        # prediction
        self.test_item = tf.placeholder(tf.int32, shape=None)
        self.test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(self.test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], tf.shape(self.test_item)[0]])
        self.pred_last = tf.argsort(tf.argsort(-self.test_logits))

    def set_vanilla_loss(self):
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def update_ewc_loss(self, ewc_lambda):
        """
        Update loss to EWC loss
        """
        self.ewc_loss = self.loss
        for v in range(len(self.variables)):
            self.ewc_loss += (ewc_lambda / 2.0) * \
                             tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),
                                                       tf.square(self.variables[v] - self.variables_prev[v])))
        self.train_op = self.optimizer.minimize(self.ewc_loss, global_step=self.global_step)

    def update_exemplar_loss(self, lambda_):
        """
        Update exemplar loss
        """
        if not self.args.use_distillation:
            train_size = tf.shape(self.input_seq)[0] - tf.shape(self.exemplar_pos)[0]
        else:
            train_size = tf.shape(self.input_seq)[0] - tf.shape(self.exemplar_logits)[0]

        # train data
        train_logits = self.logits[:train_size]
        train_labels = self.labels[:train_size]
        self.exemp_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=train_logits))

        # exemplar data
        exemplar_logits = self.logits[train_size:]
        if not self.args.use_distillation:
            # one-hot label
            exemplar_logits = exemplar_logits[:, :self.max_item_pre]
            indices = self.exemplar_pos - 1
            exemplar_labels = tf.one_hot(indices, self.max_item_pre)
            self.exemp_loss += lambda_ * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=exemplar_labels, logits=exemplar_logits))
        else:
            # logits-matching
            exemplar_logits = exemplar_logits[:, :tf.shape(self.exemplar_logits)[1]]
            exemplar_logits = exemplar_logits
            exemplar_labels = tf.nn.softmax(self.exemplar_logits)
            self.exemp_loss += lambda_ * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=exemplar_labels, logits=exemplar_logits))
        self.train_op = self.optimizer.minimize(self.exemp_loss, global_step=self.global_step)

    def compute_fisher(self, sess, data, batch_size, max_item, dropout_rate):
        """
        Compute Fisher information for each parameter
        :param sess: TensorFlow session
        :param data: selected data to compute fisher
        :param batch_size: batch size to compute fisher
        :param max_item: current period item number
        :param dropout_rate: dropout_rate
        """
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.variables)):
            self.F_accum.append(np.zeros(self.variables[v].get_shape().as_list()))

        # select random input session
        fisher_sampler = Sampler(self.args, [], batch_size=batch_size)
        fisher_sampler.prepare_data()
        fisher_sampler.add_exemplar(data)
        fisher_sampler.shuffle_data()
        batch_num = fisher_sampler.batch_num
        for _ in tqdm.tqdm(range(batch_num), desc='Computing fisher', ncols=70, leave=False):
            seq, pos = fisher_sampler.sampler()
            for i in range(len(seq)):
                # compute first-order derivatives
                input_seq = seq[i].reshape((1, -1))
                input_pos = pos[i]
                ders = sess.run(self.gradient,
                                feed_dict={self.input_seq: input_seq,
                                           self.pos: input_pos,
                                           self.max_item: max_item,
                                           self.is_training: False,
                                           self.dropout_rate: dropout_rate})
                slice = ders[1]
                dense = np.zeros(slice.dense_shape)
                for t in range(len(slice.indices)):
                    dense[slice.indices[t]] = slice.values[t]
                ders[1] = dense
                # square the derivatives and add to total
                for v in range(len(self.F_accum)):
                    self.F_accum[v] += np.square(ders[v])
        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= len(data)

    def predict(self, sess, seq, item_idx):
        """
        Predict next item
        :param sess: TensorFlow session
        :param seq: input item sequence (session)
        :param item_idx: candidate item index
        :return: rank of candidate items
        """
        return sess.run(self.pred_last, {self.input_seq: seq,
                                         self.test_item: item_idx,
                                         self.is_training: False,
                                         self.dropout_rate: self.args.dropout_rate})
