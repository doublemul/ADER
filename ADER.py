#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : ADER.py
# @Description  : ADER model file.
# The implemention of self-attentive recommender is modified based on https://github.com/kang205/SASRec

from modules import *
import tensorflow.compat.v1 as tf
import tqdm


class Ader():
    def __init__(self, item_num, args, reuse=None):
        self.args = args
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=None)
        self.exemplar_logits = tf.placeholder(tf.float32, shape=(None, None))
        self.exemplar_pos = tf.placeholder(tf.int32, shape=None)
        self.max_item = tf.placeholder(tf.int32, shape=())
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

        # define loss
        seq_emb = tf.reshape(self.rep, [tf.shape(self.input_seq)[0], args.hidden_units])
        indices = pos - 1
        self.labels = tf.one_hot(indices, self.max_item)
        item_emb = tf.nn.embedding_lookup(item_emb_table, tf.range(1, self.max_item + 1))
        self.logits = tf.matmul(seq_emb, tf.transpose(item_emb))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

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

    def update_loss(self, lambda_):
        """ Update exemplar loss
        """
        # find the number of train data from current cycle
        if self.args.disable_distillation:
            train_size = tf.shape(self.input_seq)[0] - tf.shape(self.exemplar_pos)[0]
        else:
            train_size = tf.shape(self.input_seq)[0] - tf.shape(self.exemplar_logits)[0]

        # training data
        train_logits = self.logits[:train_size]
        train_labels = self.labels[:train_size]
        self.exemp_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=train_logits))

        # exemplar data
        exemplar_logits = self.logits[train_size:]

        if self.args.disable_distillation:
            # one-hot label
            indices = self.exemplar_pos - 1
            exemplar_labels = tf.one_hot(indices, self.max_item)
            self.exemp_loss += lambda_ * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=exemplar_labels, logits=exemplar_logits))
        else:
            # logits-matching
            exemplar_logits = exemplar_logits[:, :tf.shape(self.exemplar_logits)[1]]
            exemplar_labels = tf.nn.softmax(self.exemplar_logits)
            self.exemp_loss += lambda_ * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=exemplar_labels, logits=exemplar_logits))
        self.train_op = self.optimizer.minimize(self.exemp_loss, global_step=self.global_step)

    def predict(self, sess, seq, item_idx):
        """ Predict next item
        :param sess: TensorFlow session
        :param seq: input item sequence (session)
        :param item_idx: candidate item index
        :return: rank of candidate items
        """
        return sess.run(self.pred_last, {self.input_seq: seq,
                                         self.test_item: item_idx,
                                         self.is_training: False,
                                         self.dropout_rate: self.args.dropout_rate})
