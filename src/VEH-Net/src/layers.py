# -*- coding: utf-8 -*-
# /usr/bin/python2
import pdb

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell
from params import Params
from zoneout import ZoneoutWrapper
from tensorflow.contrib.layers import batch_norm

'''
attention weights from https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
W_u^Q.shape:    (2 * attn_size, attn_size)
W_u^P.shape:    (2 * attn_size, attn_size)
W_v^P.shape:    (attn_size, attn_size)
W_g.shape:      (4 * attn_size, 4 * attn_size)
W_h^P.shape:    (2 * attn_size, attn_size)
W_v^Phat.shape: (2 * attn_size, attn_size)
W_h^a.shape:    (2 * attn_size, attn_size)
W_v^Q.shape:    (attn_size, attn_size)
'''


def get_attn_params(attn_size, initializer=tf.truncated_normal_initializer):
    '''
    Args:
        attn_size: the size of attention specified in https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
        initializer: the author of the original paper used gaussian initialization however I found xavier converge faster

    Returns:
        params: A collection of parameters used throughout the layers
    '''

    eye_constant = np.eye(2 * attn_size)
    eye_constant_initializer = tf.constant_initializer(eye_constant)
    with tf.variable_scope("attention_weights"):
        params = {
            # video_enconding_att
            # "W_ve": tf.get_variable("W_ve", dtype=tf.float32, shape=(1, Params.dim_image, 2 * attn_size),
            #                         initializer=initializer()),

            "W_v": tf.get_variable("W_v", dtype=tf.float32, shape=(1, Params.max_sentence_len),
                                   initializer=initializer()),
            "W_s": tf.get_variable("W_s", dtype=tf.float32, shape=(Params.max_sentence_len, Params.max_sentence_len),
                                   initializer=initializer()),
            "v_interaction": tf.get_variable("v_interaction", dtype=tf.float32, shape=(Params.max_sentence_len, 1),
                                             initializer=initializer()),

            "ga_W_v": tf.get_variable("ga_W_v", dtype=tf.float32, shape=(2 * attn_size, attn_size),
                                      initializer=initializer()),
            "ga_W_s": tf.get_variable("ga_W_s", dtype=tf.float32, shape=(2 * attn_size, attn_size),
                                      initializer=initializer()),
            "ga_v_interaction": tf.get_variable("ga_v_interaction", dtype=tf.float32, shape=attn_size,
                                                initializer=initializer()),

            "ga_W_fuse": tf.get_variable("ga_W_fuse", dtype=tf.float32, shape=(4 * attn_size, 2 * attn_size),
                                         initializer=initializer()),
            "ga_b_fuse": tf.get_variable("ga_b_fuse", dtype=tf.float32, shape=(2 * attn_size),
                                         initializer=initializer()),

            # Store layers weight & biases
            'h1': tf.get_variable("h1", dtype=tf.float32, shape=(attn_size, attn_size),
                                  initializer=initializer()),
            'h2': tf.get_variable("h2", dtype=tf.float32, shape=(attn_size, attn_size),
                                  initializer=initializer()),
            'out_h': tf.get_variable("out_h", dtype=tf.float32, shape=(attn_size, 1),
                                     initializer=initializer()),
            'b1': tf.get_variable("b1", shape=attn_size, dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'b2': tf.get_variable("b2", shape=attn_size, dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'out_b': tf.get_variable("out_b", shape=1, dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer()),

            "W_h_P": tf.get_variable("W_h_P", dtype=tf.float32, shape=(2 * attn_size, attn_size),
                                     initializer=initializer()),
            "W_h_a": tf.get_variable("W_h_a", dtype=tf.float32, shape=(2 * attn_size, attn_size),
                                     initializer=initializer()),
            "v": tf.get_variable("v", dtype=tf.float32, shape=attn_size, initializer=initializer()),

            "w_spool": tf.get_variable("w_spool", dtype=tf.float32, shape=(1, 2 * attn_size, 2 * attn_size))

        }
        return params


def video_encoding_att(inputs, video_len, weight, scope="video_encoding"):
    with tf.variable_scope(scope):
        output = tf.nn.conv1d(inputs, weight, stride=1, padding='VALID')
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(video_len, maxlen=Params.max_video_len), tf.int32), tf.float32)
        attn = tf.expand_dims(avg_attn, -1)
        output = attn * output
    return output


def mfb_vs_fts(sentence_fts, video_fts, sentence_len, params, scope='sentence_video_attention'):
    with tf.variable_scope(scope):
        Weights = params
        attended_fts_list = []
        for i in range(Params.max_video_len):
            with tf.variable_scope(scope):
                if i > 0: tf.get_variable_scope().reuse_variables()
                # pdb.set_trace()
                video_slice = tf.slice(video_fts, begin=[0, i, 0], size=[-1, 1, -1])
                attention_inputs = [video_slice, sentence_fts]
                vs_ft = attention_mfb(attention_inputs, Params.attn_size, Weights, memory_len=sentence_len,
                                      scope="attention_mfb")
                # pdb.set_trace()
                vs_ft = tf.reduce_sum(vs_ft, [-1])
                vs_ft = tf.multiply(tf.div(vs_ft, tf.add(tf.abs(vs_ft), 1e-10)), tf.sqrt(tf.add(tf.abs(vs_ft), 1e-10)))
                attended_fts_list.append(vs_ft)

        attended_sentence_fts = tf.transpose(tf.stack(attended_fts_list), [1, 0, 2])
        # fused_video_sentence_fts = video_fts + attended_sentence_fts
        fused_video_sentence_fts = attended_sentence_fts

        return fused_video_sentence_fts


def attention_mfb(inputs, size, weights, memory_len=None, scope="attention_mfb", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs_ = []
        weights, v = weights
        # pdb.set_trace()
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            inp = tf.transpose(inp, [0, 2, 1])
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            # if w is None:
            #     w = tf.get_variable("w_%d" % i, dtype=tf.float32, shape=[shapes[-1], 2 * Params.attn_size],
            #                         initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        # outputs = sum(outputs_)
        # pdb.set_trace()
        if memory_len is not None:
            outputs_[1] = mask_mfb_attn_score(outputs_[1], memory_len)
        outputs = tf.multiply(outputs_[0], outputs_[1])
        if Params.bias:
            b = tf.get_variable("b", shape=outputs.shape[-1], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            outputs += b
        outputs_shapes = outputs.shape.as_list()
        outputs = tf.reshape(outputs, [-1, outputs_shapes[-1]])
        vs_ft = tf.matmul(outputs, v)
        vs_ft = tf.reshape(vs_ft, [outputs_shapes[0], outputs_shapes[1], -1])
        return vs_ft


def mask_mfb_attn_score(seq_fts, memory_sequence_length):
    # pdb.set_trace()
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen=Params.max_sentence_len)
    score_mask_zeros = tf.zeros([Params.batch_size, Params.max_sentence_len], tf.float32)
    score_mask_values = tf.ones_like(score_mask_zeros)
    seq_mask = tf.where(score_mask, score_mask_values, score_mask_zeros)
    seq_mask = tf.expand_dims(seq_mask, 1)
    seq_fts_masked = seq_mask * seq_fts
    return seq_fts_masked

def Caption_Generator(img, caption_placeholder, mask, dim_in, dim_embed, dim_hidden, n_lstm_steps, n_words, init_b):
    num_filters = 5
    img = tf.squeeze(tf.layers.conv1d(img, 2*Params.attn_size, num_filters, strides=1, padding='valid', name='convcg'))
    word_embedding = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='word_embedding')
    embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
    lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
    img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
    img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')
    word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
    word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
    # getting an initial LSTM embedding from our image_imbedding
    image_embedding = tf.matmul(img, img_embedding) + img_embedding_bias

    # setting initial state of our LSTM
    state = lstm.zero_state(Params.batch_size, dtype=tf.float32)

    total_loss = 0.0
    with tf.variable_scope("RNN"):
        for i in range(n_lstm_steps):
            if i > 0:
                # if this isn’t the first iteration of our LSTM we need to get the word_embedding corresponding
                # to the (i-1)th word in our caption
                with tf.device("/cpu:0"):
                    current_embedding = tf.nn.embedding_lookup(word_embedding,
                                                               caption_placeholder[:, i - 1]) + embedding_bias
            else:
                # if this is the first iteration of our LSTM we utilize the embedded image as our input
                current_embedding = image_embedding
            if i > 0:
                # allows us to reuse the LSTM tensor variable on each iteration
                tf.get_variable_scope().reuse_variables()

            out, state = lstm(current_embedding, state)

            if i > 0:
                # get the one-hot representation of the next word in our caption
                labels = tf.expand_dims(caption_placeholder[:, i], 1)
                ix_range = tf.range(0, Params.batch_size, 1)
                ixs = tf.expand_dims(ix_range, 1)
                concat = tf.concat([ixs, labels], 1)
                onehot = tf.sparse_to_dense(
                    concat, tf.stack([Params.batch_size, n_words]), 1.0, 0.0)

                # perform a softmax classification to generate the next word in the caption
                logit = tf.matmul(out, word_encoding) + word_encoding_bias
                xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
                xentropy = xentropy * mask[:, i]

                loss = tf.reduce_sum(xentropy)
                total_loss += loss

        total_loss = total_loss / tf.reduce_sum(mask[:, 1:])
        return total_loss


def global_fts_att(params, inputs_vs, inputs, video_len, scope):
    with tf.variable_scope(scope):
        Weights, W_fuse, b_fuse = params
        # Weights = params
        attention_list = []
        attended_fts_list = []
        for i in range(Params.max_video_len):
            with tf.variable_scope(scope):
                if i > 0: tf.get_variable_scope().reuse_variables()
                # pdb.set_trace()
                video_slice = tf.slice(inputs, begin=[0, i, 0], size=[-1, 1, -1])
                attention_inputs = [video_slice, inputs_vs]
                attention_weights = attention(attention_inputs, Params.attn_size, Weights, memory_len=video_len,
                                              scope="attention")
                # pdb.set_trace()
                extended_attention_weights = tf.expand_dims(attention_weights, -1)
                attended_fts = tf.reduce_sum(extended_attention_weights * inputs_vs, 1)
                attention_list.append(attention_weights)
                attended_fts_list.append(attended_fts)
        # pdb.set_trace()
        attended_vs_fts = tf.transpose(tf.stack(attended_fts_list), [1, 0, 2])
        concat_vs_fts = tf.concat([inputs, attended_vs_fts], -1)
        fused_video_sentence_fts = tf.nn.relu(
            tf.matmul(tf.reshape(concat_vs_fts, [-1, 4 * Params.attn_size]), W_fuse) + b_fuse)
        glatt_video_sentence_fts = tf.reshape(fused_video_sentence_fts, [Params.batch_size, Params.max_video_len, -1])
        vs_fts_attentions = tf.transpose(tf.stack(attention_list), [1, 0, 2])
        # pdb.set_trace()
        return glatt_video_sentence_fts, vs_fts_attentions


# def mul_vae_loss(glatt_vs_fts, ave_sent, video_len,scope):
#     with tf.variable_scope(scope):
#         #pdb.set_trace()
#         num_filters = 2 * Params.attn_size
#         kernel_size = 1
#         conv2 = tf.layers.conv1d(glatt_vs_fts, num_filters, kernel_size, strides=1, padding='valid', name='conv21')
#         con_senfts = tf.nn.relu(conv2)
#         con2_senfts = tf.layers.conv1d(con_senfts, num_filters, kernel_size, strides=1, padding='valid', name='conv22')
#         con2_senfts_att = tf.nn.softmax(tf.reduce_sum(con2_senfts, axis=2), axis=1)
#         con2_senfts_att = tf.expand_dims(con2_senfts_att, -1)
#         memory_attn = con2_senfts_att * glatt_vs_fts
#         memory_conavg_attn, _ = avg_fts_pooling(memory_attn, memory_len=video_len)
#         loss = tf.losses.mean_squared_error(memory_conavg_attn, ave_sent)
#         return loss


# def construct_noWg_graph(inputs, video_len, scope):
#     with tf.variable_scope(scope):
#         G_pre = tf.matmul(inputs, tf.transpose(inputs, [0, 2, 1]))  # (batch_size,N,N)
#         G, G_softmax, G_softmax_I = mask_att_score_extend_matrix_normalize(G_pre, video_len)
#         return G_pre, G, G_softmax, G_softmax_I
#
#


def mask_attn_score_extend_matrix(score, memory_sequence_length, score_mask_value=-1e8):
    # score (b,max_len,max_len) (20,210,210)
    # memory_sequence_length (b) (20)

    score_mask = tf.cast(tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]),
                         tf.int32)  # (b,max_len) (20,210)
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), [1, 1, 1])  # (b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask, [0, 2, 1])  # (b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask, score_mask_transpose)  # (b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix, tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(score)

    return tf.where(score_mask_matrix, score, score_mask_values)


def mask_graph(inputs, video_len, score_mask_value=0.0):
    score_mask = tf.cast(tf.sequence_mask(video_len, maxlen=inputs.shape[1]), tf.int32)
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), [1, 1, 1])  # (b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask, [0, 2, 1])  # (b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask, score_mask_transpose)  # (b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix, tf.bool)
    score_mask_values = score_mask_value * tf.ones_like(inputs)

    return tf.where(score_mask_matrix, inputs, score_mask_values)


def mask_att_score_extend_matrix_normalize(score, memory_sequence_length):
    score_mask = tf.cast(tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1]),
                         tf.int32)  # (b,max_len) (20,210)
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), [1, 1, 1])  # (b,max_len,1) (20,210,1)
    score_mask_transpose = tf.transpose(score_mask, [0, 2, 1])  # (b,1,max_len) (20,1,210)
    score_mask_matrix = tf.matmul(score_mask, score_mask_transpose)  # (b,max_len,max_len) (20,210,210)
    score_mask_matrix = tf.cast(score_mask_matrix, tf.bool)
    score_mask_values = 0.0 * tf.ones_like(score)
    transfer_score = tf.where(score_mask_matrix, score, score_mask_values)

    score_mask_values_row_sum = tf.reduce_sum(transfer_score, 2)  # (20,210)
    score_mask_values_row_sum_extend = tf.tile(tf.expand_dims(score_mask_values_row_sum, 2),
                                               [1, 1, Params.max_video_len])  # (20,210,210)
    score_normalize = tf.div(transfer_score, score_mask_values_row_sum_extend + 1e-8)

    score_mask_values = -1e8 * tf.ones_like(score_normalize)
    score_normalize_softmax_pre = tf.where(score_mask_matrix, score_normalize, score_mask_values)
    score_normalize_softmax = mask_graph(tf.nn.softmax(score_normalize_softmax_pre * Params.gcn_softmax_scale, dim=2),
                                         memory_sequence_length)

    idx = tf.convert_to_tensor(np.eye(Params.max_video_len, dtype=np.float32))
    score_normalize_softmax_I_pre = tf.nn.softmax(score_normalize_softmax_pre * Params.gcn_softmax_scale, dim=2) + idx
    score_normalize_softmax_I = mask_graph(score_normalize_softmax_I_pre, memory_sequence_length)

    return score_normalize, score_normalize_softmax, score_normalize_softmax_I


def encoding(word, char, word_embeddings, char_embeddings, scope="embedding"):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, word)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, char)
        return word_encoding, char_encoding


def apply_dropout(inputs, size=None, is_training=True):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if Params.dropout is None and Params.zoneout is None:
        return inputs
    if Params.zoneout is not None:
        return ZoneoutWrapper(inputs, state_zoneout_prob=Params.zoneout, is_training=is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                             output_keep_prob=1 - Params.dropout,
                                             # variational_recurrent = True,
                                             # input_size = size,
                                             dtype=tf.float32)
    else:
        return inputs


def bidirectional_GRU(inputs, inputs_len, cell=None, cell_fn=tf.contrib.rnn.GRUCell, units=Params.attn_size, layers=1,
                      scope="Bidirectional_GRU", output=0, is_training=True, reuse=None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs, (shapes[0] * shapes[1], shapes[2], -1))
                inputs_len = tf.reshape(inputs_len, (shapes[0] * shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell(
                    [apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units, is_training=is_training)
                     for i in range(layers)])
                cell_bw = MultiRNNCell(
                    [apply_dropout(cell_fn(units), size=inputs.shape[-1] if i == 0 else units, is_training=is_training)
                     for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(cell_fn(units), size=inputs.shape[-1], is_training=is_training) for _
                                    in range(2)]

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                          sequence_length=inputs_len,
                                                          dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states, 1), (Params.batch_size, shapes[1], 2 * units))


def sequence_prototypes_net(video, video_len, forward_position_list, sentence_state, cell_f, params, weights_mlp, bias_mlp,
                            scope=None):
    weights_p = params

    video_shape = np.shape(video)
    zeros_video_padding = tf.zeros([video_shape[0], 1, video_shape[2]], tf.float32)
    expand_video = tf.concat([video, zeros_video_padding], 1)
    inputs = [expand_video, sentence_state]

    p_logits_list = []
    p_logits_list_new = []
    for i in range(Params.max_choose_len + 1):
        with tf.variable_scope('sequence_attention'):
            #pdb.set_trace()
            if i > 0: tf.get_variable_scope().reuse_variables()

            if i == 0:
                f_state = sentence_state
                f_inputs = inputs
                f_direction = 'start'
            else:
                f_direction = 'forward'
                f_inputs = [expand_video, f_state]
            with tf.variable_scope('forward'):
                # pdb.set_trace()
                p_logits = \
                    mfb_attention_with_padding(f_inputs, Params.attn_size,
                                               weights_p, weights_mlp, bias_mlp,
                                               memory_len=video_len,
                                               current_position=forward_position_list[:, i],
                                               direction=f_direction,
                                               scope="attention_f")
                # idx_list = tf.contrib.framework.argsort(p_logits_new, axis=1, direction='ASCENDING')
                idx_list = tf.contrib.framework.argsort(p_logits, axis=1, direction='ASCENDING')
                predicted_position = idx_list[:, -1]
                p_logits_list_new.append(predicted_position)
                # f_inputs = [expand_video, f_state]
                p_logits_list.append(p_logits)
                if i < 5:
                    current_position = forward_position_list[:, i+1]
                else:
                    current_position = forward_position_list[:, 5]
                current_position = tf.transpose(tf.stack([current_position]), [1, 0])
                prethumb = tf.squeeze(tf.batch_gather(expand_video, current_position))
                _, f_state = cell_f(prethumb, f_state)

                # scores = tf.expand_dims(p_logits, -1)
                # pdb.set_trace()
                # new_expand_video = scores * expand_video
                # new_video_state = new_expand_video + expand_video
                #pdb.set_trace()
                # attention_pool = tf.reduce_sum(scores * expand_video, 1)
                # _, f_state = cell_f(attention_pool, f_state)
                # f_inputs = [new_video_state, f_state]
    # return tf.stack(p_logits_list)
    return tf.stack(p_logits_list), tf.transpose(tf.stack(p_logits_list_new), [1, 0]), expand_video


def inference_sequence_prototypes_net(i, video, video_len, forward_position, sentence_state, cell_f, inference_f_state,
                                      params, weights_mlp, bias_mlp, scope=None):
    weights_p = params

    video_shape = np.shape(video)
    zeros_video_padding = tf.zeros([video_shape[0], 1, video_shape[2]], tf.float32)
    expand_video = tf.concat([video, zeros_video_padding], 1)

    with tf.variable_scope('sequence_attention'):
        def true_proc():
            f_inputs = [expand_video, inference_f_state]
            # f_inputs = [inference_video_state, inference_f_state]
            f_state = inference_f_state
            return f_inputs, f_state

        def false_proc():
            f_inputs = [expand_video, sentence_state]
            f_state = sentence_state
            return f_inputs, f_state

        f_inputs, f_state = tf.cond(tf.greater(i, tf.constant(0)), true_fn=true_proc, false_fn=false_proc)

        with tf.variable_scope('forward'):
            #pdb.set_trace()
            p_logits = mfb_attention_with_padding(f_inputs, Params.attn_size,
                                                  weights_p, weights_mlp, bias_mlp,
                                                  memory_len=video_len,
                                                  current_position=forward_position,
                                                  direction='forward',
                                                  scope="attention_f")
            # scores = tf.expand_dims(p_logits, -1)
            # attention_pool = tf.reduce_sum(scores * expand_video, 1)
            idx_list = tf.contrib.framework.argsort(p_logits, axis=1, direction='ASCENDING')
            current_position = idx_list[:, -1]
            current_position = tf.transpose(tf.stack([current_position]), [1, 0])
            prethumb = tf.squeeze(tf.batch_gather(expand_video, current_position))
            _, f_o_state = cell_f(prethumb, f_state)


    return p_logits, f_o_state, tf.greater(i, tf.constant(0))


def mfb_attention_with_padding(inputs, units, weights, weights_mlp, bias_mlp, scope, memory_len=None,
                               current_position=None, direction=None,
                               reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # pdb.set_trace()
        outputs_ = []
        weights, v = weights
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d" % i, dtype=tf.float32, shape=[shapes[-1], Params.attn_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        # pdb.set_trace()
        outputs = tf.multiply(outputs_[0], outputs_[1])
        outputs = tf.multiply(tf.div(outputs, tf.add(tf.abs(outputs), 1e-10)), tf.sqrt(tf.add(tf.abs(outputs), 1e-10)))
        # if Params.bias:
        #     b = tf.get_variable("b", shape=outputs.shape[-1], dtype=tf.float32,
        #                         initializer=tf.contrib.layers.xavier_initializer())
        #     outputs += b
        scores = tf.reduce_sum(multilayer_perceptron(outputs, weights_mlp, bias_mlp), [-1])
        if memory_len is not None:
            scores = mask_attn_score_with_padding_with_position(scores, memory_len, current_position, direction)
        return tf.nn.softmax(scores)  # all attention output is softmaxed now


def multilayer_perceptron(X, weights, bias, scope="MLP"):
    # pdb.set_trace()
    with tf.variable_scope(scope):
        shapes = X.shape.as_list()
        X = tf.reshape(X, (-1, shapes[-1]))
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[0]), bias[0]))  # Hidden layer with relu activation
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights[1]), bias[1]))  # Hidden layer with Relu activation
        output = tf.matmul(layer2, weights[2]) + bias[2]
        output = tf.reshape(output, (shapes[0], shapes[1], -1))
    return output


def attention_rnn(inputs, inputs_len, units, attn_cell, bidirection=True, scope="gated_attention_rnn",
                  is_training=True):
    with tf.variable_scope(scope):
        if bidirection:
            outputs = bidirectional_GRU(inputs,
                                        inputs_len,
                                        cell=attn_cell,
                                        scope=scope + "_bidirectional",
                                        output=0,
                                        is_training=is_training)
        else:
            outputs, _ = tf.nn.dynamic_rnn(attn_cell, inputs,
                                           sequence_length=inputs_len,
                                           dtype=tf.float32)
        return outputs


def sentence_pooling(memory, units, weights, memory_len=None, scope="sentence_pooling"):
    with tf.variable_scope(scope):
        shapes = memory.get_shape().as_list()
        V_r = tf.get_variable("sentence_param", shape=(Params.max_sentence_len, units),
                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        inputs_ = [memory, V_r]
        attn = attention(inputs_, units, weights, memory_len=memory_len, scope="sentence_attention_pooling")
        attn = tf.expand_dims(attn, -1)
        return tf.reduce_sum(attn * memory, 1)


def avg_sentence_pooling(memory, memory_len=None, scope="sentence_pooling"):
    with tf.variable_scope(scope):
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(memory_len, maxlen=Params.max_sentence_len), tf.int32), tf.float32)
        avg_attn = tf.div(avg_attn,
                          tf.cast(tf.tile(tf.expand_dims(memory_len, 1), [1, Params.max_sentence_len]), tf.float32))
        attn = tf.expand_dims(avg_attn, -1)
        return tf.reduce_sum(attn * memory, 1)


def avg_fts_pooling(memory, memory_len=None, scope="fts_pooling"):
    with tf.variable_scope(scope):
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(memory_len, maxlen=Params.max_video_len), tf.int32), tf.float32)
        avg_attn = tf.div(avg_attn,
                          tf.cast(tf.tile(tf.expand_dims(memory_len, 1), [1, Params.max_video_len]), tf.float32))
        attn = tf.expand_dims(avg_attn, -1)
        return tf.reduce_sum(attn * memory, 1), avg_attn


def conv1d_sentence_pooling(memory, units, memory_len=None, scope="csentence_pooling"):
    with tf.variable_scope(scope):
        # pdb.set_trace()
        num_filters = 2 * Params.attn_size
        kernel_size = 1
        conv2 = tf.layers.conv1d(memory, num_filters, kernel_size, strides=1, padding='valid', name='conv21')
        con_senfts = tf.nn.relu(conv2)
        con2_senfts = tf.layers.conv1d(con_senfts, num_filters, kernel_size, strides=1, padding='valid', name='conv22')
        con2_senfts_att = tf.nn.softmax(tf.reduce_sum(con2_senfts, axis=2), axis=1)
        con2_senfts_att = tf.expand_dims(con2_senfts_att, -1)
        memory_attn = con2_senfts_att * memory
        memory_conavg_attn = avg_sentence_pooling(memory_attn, memory_len=memory_len)
        return memory_conavg_attn, con2_senfts_att


def gated_attention(memory, inputs, states, units, params, self_matching=False, memory_len=None,
                    scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states, (Params.batch_size, Params.attn_size))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len=memory_len)
        save_attention = scores
        scores = tf.expand_dims(scores, -1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs, attention_pool), axis=1)
        g_t = tf.sigmoid(tf.matmul(inputs, W_g))
        return g_t * inputs


def mask_attn_score(score, memory_sequence_length, score_mask_value=-1e8):
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def mask_attn_score_with_padding(score, memory_sequence_length, score_mask_value=-1e8):
    score_mask_pre = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1] - 1)
    score_mask_padding = tf.ones([score.shape[0], 1], tf.int32)
    score_mask_padding = tf.cast(score_mask_padding, tf.bool)
    score_mask = tf.concat([score_mask_pre, score_mask_padding], 1)
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def mask_attn_score_with_padding_with_position(score, memory_sequence_length, current_sequence_position, direction,
                                               score_mask_value=-1e8):
    # pdb.set_trace()
    score_mask_all = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1] - 1)

    if direction == 'forward':
        current_mask_pre = tf.sequence_mask(current_sequence_position + 1, maxlen=score.shape[1] - 1)
        current_mask_transfer = 1 - tf.cast(current_mask_pre, tf.int32)
    elif direction == 'backward':
        current_mask_pre = tf.sequence_mask(current_sequence_position, maxlen=score.shape[1] - 1)
        current_mask_transfer = tf.cast(current_mask_pre, tf.int32)
    elif direction == 'start':
        current_mask_pre = tf.sequence_mask(score.shape[1] - 1, maxlen=score.shape[1] - 1)  # all true
        # current_mask_pre = tf.sequence_mask([tf.cast(score.shape[1]-1,tf.int32)], maxlen=score.shape[1] - 1)
        current_mask_transfer = tf.cast(current_mask_pre, tf.int32)

    score_mask_pre = tf.cast(tf.cast(score_mask_all, tf.int32) * current_mask_transfer, tf.bool)

    score_mask_padding = tf.ones([score.shape[0], 1], tf.int32)
    score_mask_padding = tf.cast(score_mask_padding, tf.bool)
    score_mask = tf.concat([score_mask_pre, score_mask_padding], 1)

    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def attention(inputs, units, weights, scope="attention", memory_len=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d" % i, dtype=tf.float32, shape=[shapes[-1], Params.attn_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape=outputs.shape[-1], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores)  # all attention output is softmaxed now


def attention_with_padding(inputs, units, weights, scope, memory_len=None, current_position=None, direction=None,
                           reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp, w) in enumerate(zip(inputs, weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d" % i, dtype=tf.float32, shape=[shapes[-1], Params.attn_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is Params.batch_size:
                outputs = tf.reshape(outputs, (shapes[0], 1, -1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0], -1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if Params.bias:
            b = tf.get_variable("b", shape=outputs.shape[-1], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score_with_padding_with_position(scores, memory_len, current_position, direction)
        return tf.nn.softmax(scores)  # all attention output is softmaxed now


def loss_choose(output_f, target_f):
    choose_probability = target_f * tf.log(output_f + 1e-8)
    loss_choose = -tf.reduce_sum(choose_probability)
    return loss_choose


def loss_diversity(vs_fts, choosed_indexes):
    # pdb.set_trace()
    choosed_vs_fts = tf.batch_gather(vs_fts, choosed_indexes)
    num = Params.max_choose_len * (Params.max_choose_len - 1)
    cos_list = []
    for i in range(Params.batch_size):
        choosed_slice = tf.reduce_sum(tf.slice(choosed_vs_fts, begin=[i, 0, 0], size=[1, Params.max_choose_len, -1]), 0)
        slice_normalized = tf.nn.l2_normalize(choosed_slice, dim=1)
        slice_prod = tf.matmul(slice_normalized, slice_normalized, adjoint_b=True)
        slice_cos_similarity = tf.div(tf.reduce_sum(1 - slice_prod), num)
        cos_list.append(slice_cos_similarity)
    return tf.reduce_sum(cos_list)


# def GAN_vae():
#


def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))


def LayerNormalization(inputs, epsilon=1e-5, scope=None):
    """ Computer LN given an input tensor. We get in an input of shape
    [N X D] and with LN we compute the mean and var for each individual
    training point across all it's hidden dimensions rather than across
    the training batch as we do in BN. This gives us a mean and var of shape
    [N X 1].
    """
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN


def BatchNormalization(inputs, is_training, scope=None):
    with tf.variable_scope(scope + 'BN') as BN:
        # BN_output = tf.cond(is_training,
        #     lambda: batch_norm(
        #         inputs, is_training=True, center=True,
        #         scale=True, activation_fn=tf.nn.relu,
        #         updates_collections=None, scope=BN),
        #     lambda: batch_norm(
        #         inputs, is_training=False, center=True,
        #         scale=True, activation_fn=tf.nn.relu,
        #         updates_collections=None, scope=BN, reuse=True))
        def true_proc():
            return batch_norm(
                inputs, is_training=True, center=True,
                scale=True, activation_fn=tf.nn.relu,
                updates_collections=None, scope=BN)

        def false_proc():
            return batch_norm(
                inputs, is_training=False, center=True,
                scale=True, activation_fn=tf.nn.relu,
                updates_collections=None, scope=BN, reuse=True)

        BN_output = tf.cond(tf.cast(is_training, tf.bool), true_fn=true_proc, false_fn=false_proc)

        return BN_output
