# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import pdb

import tensorflow as tf
from tqdm import tqdm
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
import numpy as np
import cPickle as pickle
from params import Params
import numpy as np


class VEHNetunsup(object):

    def __init__(self, bias_init_vector, word_emb_init, is_training=True):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()

        self.video = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len,
                                                 Params.dim_image])  # (batch_size, timestep, dim)
        self.sentence_index = tf.placeholder(tf.int32,
                                             [Params.batch_size, Params.max_sentence_len])  # (batch_size, timestep)
        self.video_w_len = tf.placeholder(tf.int32, [Params.batch_size, ])  # (batch_size,)
        self.sentence_w_len = tf.placeholder(tf.int32, [Params.batch_size, ])  # (batch_size,)
        self.indices = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len])  # (batch_size, timestep+1)
        # caption generation
        # img = tf.placeholder(tf.float32, [Params.batch_size, Params.attn_size*2])
        self.sentence_matrix = tf.placeholder(tf.int32, [Params.batch_size, Params.max_sentence_len + 2])
        self.sentence_mask = tf.placeholder(tf.float32, [Params.batch_size, Params.max_sentence_len + 2])
        self.dim_in = 2*Params.attn_size
        self.dim_embed = 2*Params.attn_size
        self.dim_hidden = 2*Params.attn_size
        self.n_lstm_steps = Params.max_sentence_len+2
        self.n_words = len(word_emb_init)
        self.bias_init = bias_init_vector

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(initial_value=word_emb_init, name='Wemb')
            sentence_emb = []
            for i in xrange(Params.max_sentence_len):
                sentence_emb.append(tf.nn.embedding_lookup(self.Wemb, self.sentence_index[:, i]))
            sentence_emb = tf.stack(sentence_emb)
            self.sentence = tf.transpose(sentence_emb, [1, 0, 2])

        self.params = get_attn_params(Params.attn_size, initializer=tf.contrib.layers.xavier_initializer)

        # video and sentence encoding
        self.video_encoding = bidirectional_GRU(self.video,
                                                self.video_w_len,
                                                cell_fn=SRUCell if Params.SRU else GRUCell,
                                                layers=Params.num_layers,
                                                scope="video_encoding",
                                                output=0,
                                                is_training=self.is_training)
        # self.video_encoding = video_encoding_att(self.video,
        #                                          self.video_w_len,
        #                                          self.params["W_ve"],
        #                                          scope="video_encoding")

        self.sentence_encoding = bidirectional_GRU(self.sentence,
                                                   self.sentence_w_len,
                                                   cell_fn=SRUCell if Params.SRU else GRUCell,
                                                   layers=Params.num_layers,
                                                   scope="sentence_encoding",
                                                   output=0,
                                                   is_training=self.is_training)

        self.cell_f = apply_dropout(GRUCell(Params.attn_size * 2), is_training=self.is_training)

        self.MFB()
        # self.similarity_graph_construction()
        # self.graph_convolution()
        self.global_ftsatt()
        self.bidirectional_readout()
        self.get_sentence_state()

        if self.is_training:
            self.current_position_list = tf.placeholder(tf.int32, [Params.batch_size, Params.max_choose_len + 1])
            self.outputs_train()
            self.caption_gene()
            self.loss_function()
        else:
            self.inference_f_state = tf.placeholder(tf.float32, [Params.batch_size, Params.attn_size * 2])
            # self.inference_video_state = tf.placeholder(tf.float32, [Params.batch_size, Params.max_video_len + 1,
            #                                                          Params.attn_size * 2])
            self.current_position = tf.placeholder(tf.int32, [Params.batch_size])
            self.inference_step = tf.placeholder(tf.int32)
            self.outputs_inference()

        total_params()

    def MFB(self):
        weights_interaction = ([self.params["W_v"], self.params["W_s"]], self.params["v_interaction"])
        self.sentece_video_interaction_fts = mfb_vs_fts(self.sentence_encoding, self.video_encoding,
                                                        self.sentence_w_len,
                                                        weights_interaction)

    # def sentence_video_interaction(self):
    #     weights_interaction = ([self.params["W_v"], self.params["W_s"]], self.params["v_interaction"]), self.params[
    #         "W_fuse"], self.params["b_fuse"]
    #     self.sentece_video_interaction_fts, self.sentence_video_attention \
    #         = attend_sentence_with_video_fts(self.sentence_encoding, self.video_encoding, self.sentence_w_len,
    #                                          weights_interaction)

    # def similarity_graph_construction(self):
    #     self.G_pre, self.G, self.G_softmax, self.G_softmax_I = construct_noWg_graph(self.sentece_video_interaction_fts,
    #                                                                                 self.video_w_len,
    #                                                                                 scope="similarity_graph_construction")
    #
    # def graph_convolution(self):
    #     params = [self.params["W_c1"], self.params["W_c2"], self.params["W_c3"], self.params["W_c4"]]
    #     inputs = self.sentece_video_interaction_fts
    #     for i in range(Params.num_gcn):
    #         inputs = GCN(self.G_softmax_I, inputs, params[i], scope="graph_convolution_" + str(i),
    #                      is_training=self.is_training)
    #     self.video_gcn_represent = inputs

    def global_ftsatt(self):
        params = ([self.params["ga_W_v"], self.params["ga_W_s"]], self.params["ga_v_interaction"]), self.params[
            "ga_W_fuse"], self.params["ga_b_fuse"]
        inputs_sv = self.sentece_video_interaction_fts
        inputs = self.video_encoding
        self.fts_gatt, self.vs_gattention = global_fts_att(params, inputs_sv, inputs, self.video_w_len,
                                                           scope="global_ftsatt")

    def bidirectional_readout(self):
        self.final_bidirectional_outputs = bidirectional_GRU(self.fts_gatt,
                                                             self.video_w_len,
                                                             cell_fn=SRUCell if Params.SRU else GRUCell,
                                                             # layers = Params.num_layers, # or 1? not specified in the original paper
                                                             scope="bidirectional_readout",
                                                             output=0,
                                                             is_training=self.is_training)

    def get_sentence_state(self):
        with tf.variable_scope("sentence_self_encoding"):
            self.sentence_state, self.sentence_attn = conv1d_sentence_pooling(self.sentence_encoding,
                                                                              units=Params.attn_size,
                                                                              memory_len=self.sentence_w_len,
                                                                              scope="sentence_pooling")

    def outputs_train(self):
        params = ([self.params["W_h_P"], self.params["W_h_a"]], self.params["v"])
        weights = self.params["h1"], self.params["h2"], self.params["out_h"]
        bias = self.params["b1"], self.params["b2"], self.params["out_b"]
        # pdb.set_trace()
        self.probability_f, self.expand_video, self.new_expand_video, self.new_probability_f = new_sequence_pointer_net(
            self.final_bidirectional_outputs,
            self.video_w_len,
            self.current_position_list,
            self.sentence_state, self.cell_f,
            params, weights,
            bias, )  # scope = "sequence_pointer_network"
        # self.choosed_vs_fts = tf.batch_gather(self.expand_video, self.choose_index_list[:, :5])

    def outputs_inference(self):
        params = ([self.params["W_h_P"], self.params["W_h_a"]], self.params["v"])
        weights = [self.params["h1"], self.params["h2"], self.params["out_h"]]
        bias = [self.params["b1"], self.params["b2"], self.params["out_b"]]
        # self.probability_f_inference, self.output_f_state, self.output_video_state, self.output_cond = \
        #     inference_sequence_pointer_net(self.inference_step, self.final_bidirectional_outputs, self.video_w_len,
        #                                    self.current_position, self.sentence_state, self.cell_f,
        #                                    self.inference_f_state,
        #                                    self.inference_video_state,
        #                                    params, weights, bias, )  # scope = "sequence_pointer_network_inference"
        self.probability_f_inference, self.output_f_state, self.output_cond = \
            new_inference_sequence_pointer_net(self.inference_step, self.final_bidirectional_outputs, self.video_w_len,
                                               self.current_position, self.sentence_state, self.cell_f,
                                               self.inference_f_state,
                                               params, weights, bias, )  # scope = "sequence_pointer_network_inference"

    def caption_gene(self):
        self.cap_loss =Caption_Generator(self.new_expand_video, self.sentence_matrix, self.sentence_mask, self.dim_in,
                                         self.dim_embed, self.dim_hidden, self.n_lstm_steps, self.n_words,
                                         self.bias_init)


    def loss_function(self):
        with tf.variable_scope("loss"):
            # self.choose_loss_f = Params.alpha_choose * loss_choose(self.probability_f, self.indices)
            tv = tf.trainable_variables()
            self.regular_loss = Params.alpha_regular * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            # self.recon_loss = Params.batch_size * Params.attn_size * 2 * mul_vae_loss(glatt_vs_fts=self.fts_gatt, ave_sent=self.sentence_state, video_len=self.video_w_len, scope="reconstruction_loss")
            # self.loss = self.choose_loss_f + self.regular_loss + self.recon_loss
            self.cap_loss = 100 * self.cap_loss
            # self.sparsity_loss = tf.square(tf.reduce_mean(tf.reduce_sum(self.new_probability_f, 1), 0)-5)
            # self.len_loss = tf.cast(tf.reduce_mean(tf.reduce_sum(self.choose_index_list, 1), 0), tf.float32)
            #pdb.set_trace()
            # self.loss = self.choose_loss_f + self.regular_loss + self.cap_loss
            # self.loss = self.regular_loss + self.cap_loss +self.sparsity_loss
            self.loss = self.regular_loss + self.cap_loss


def debug():
    word_emb_init = np.array(np.load(open(Params.word_fts_path)).tolist(), np.float32)
    model = new_RGCN_sequence(word_emb_init, is_training=False)
    print("Built model")

# debug()
