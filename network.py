import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq
from config import *
import numpy as np


sequence_length = np.array([max_length] * batch_size, dtype=np.int32)


def attention_network(X_encoder, X_decoder, encoder_length, z_encoder, z_decoder, dropout_prob):
    with tf.variable_scope('attention_scope', reuse=tf.AUTO_REUSE):
        embeddings = tf.Variable(tf.random_normal(
            [signal_size, embedding_size],
            mean=0.0, stddev=1.0, dtype=tf.float32), name="embedding")

        def lstm_cell(hidden_size, cell_id=0):
            cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='attn_cell%d' % cell_id)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_prob)
            return cell

        signal_encoder  = tf.nn.embedding_lookup(embeddings, z_encoder)  # 将z用word embedding的方式进行编码, 可以被训练
        signal_decoder  = tf.nn.embedding_lookup(embeddings, z_decoder)  # 将z用word embedding的方式进行编码, 可以被训练
        X_encoder_input = tf.concat([signal_encoder, X_encoder], axis=2)             # 将X和z进行拼接
        X_decoder_input = tf.concat([signal_decoder, X_decoder], axis=2)             # 将X和z进行拼接

        fw_cell = lstm_cell(hidden_size, 0)
        bw_cell = lstm_cell(hidden_size, 1)
        fw_zero = fw_cell.zero_state(batch_size, tf.float32)
        bw_zero = bw_cell.zero_state(batch_size, tf.float32)
        encoder_output, encoder_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X_encoder_input,
                                             sequence_length=encoder_length,
                                               initial_state_fw=fw_zero, initial_state_bw=bw_zero,dtype=tf.float32)

        attn_context = tf.concat(encoder_output, axis=2)
        attn_mech = seq2seq.BahdanauAttention(hidden_size * 2, memory=attn_context, memory_sequence_length=encoder_length)
        cells = rnn.MultiRNNCell([lstm_cell(hidden_size, 2), lstm_cell(hidden_size, 3), lstm_cell(hidden_size, 4)], state_is_tuple=True)
        attn_cell = seq2seq.AttentionWrapper(cell=cells,
                                        attention_mechanism=attn_mech,
                                        attention_layer_size=hidden_size, alignment_history=True)

        # decode_cell_in = lstm_cell(hidden_size, 5)
        # decode_cell_out = lstm_cell(hidden_size, 6)
        # decode_cells = rnn.MultiRNNCell([decode_cell_in, attn_cell, decode_cell_out], state_is_tuple=True)

        # densen = tf.layers.Dense(vector_size)
        helper = seq2seq.TrainingHelper(inputs=X_decoder_input, sequence_length=sequence_length)
        decoder_init_state = attn_cell.zero_state(batch_size, tf.float32)
        decoder = seq2seq.BasicDecoder(cell=attn_cell, helper=helper, initial_state=decoder_init_state)
        decoder_output, decoder_state, _ = seq2seq.dynamic_decode(decoder=decoder)

        # Full Connection层1
        w1 = tf.get_variable("fcn_w_1", [1, hidden_size, vector_size])
        b1 = tf.get_variable("fcn_b_1", [vector_size])

        # Full Connection层2
        w2 = tf.get_variable("fcn_w_2", [1, vector_size, vocab_size])
        b2 = tf.get_variable("fcn_b_2", [vocab_size])

        w1_tile = tf.tile(w1, [batch_size, 1, 1])
        w2_tile = tf.tile(w2, [batch_size, 1, 1])
        w1_output = tf.nn.xw_plus_b(decoder_output[0], w1_tile, b1)
        output = tf.nn.xw_plus_b(w1_output, w2_tile, b2, name="output")

        # output = tf.nn.softmax(output)

        return output, decoder_state


def network(X, z, dropout_prob):
    embeddings = tf.Variable(tf.random_normal(
        [signal_size, embedding_size],
        mean=0.0, stddev=1.0, dtype=tf.float32), name="embedding")
    signal = tf.nn.embedding_lookup(embeddings, z)  # 将z用word embedding的方式进行编码, 可以被训练
    X = tf.concat([signal, X], axis=2)  # 将X和z进行拼接

    def lstm_cell(hidden_size, cell_id=0):
        # LSTM细胞生成器
        cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name='cell%d' % cell_id)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_prob)
        return cell

    # 3层lstm
    cells = rnn.MultiRNNCell([lstm_cell(hidden_size, 0),
                              lstm_cell(hidden_size, 1),
                              lstm_cell(hidden_size, 2)], state_is_tuple=True)
    initial_state = cells.zero_state(batch_size, tf.float32)
    cell_output, cell_state = tf.nn.dynamic_rnn(cells, X, initial_state=initial_state, dtype=tf.float32)

    # 2个全连接层
    with tf.variable_scope('fcn1', reuse=tf.AUTO_REUSE):
        # Full Connection层1
        w1 = tf.get_variable("fcn_w_1", [1, hidden_size, vector_size])
        b1 = tf.get_variable("fcn_b_1", [vector_size])

    with tf.variable_scope('fcn2', reuse=tf.AUTO_REUSE):
        # Full Connection层2
        w2 = tf.get_variable("fcn_w_2", [1, vector_size, vocab_size])
        b2 = tf.get_variable("fcn_b_2", [vocab_size])

    w1_tile = tf.tile(w1, [batch_size, 1, 1])
    w2_tile = tf.tile(w2, [batch_size, 1, 1])
    w1_output = tf.nn.xw_plus_b(cell_output, w1_tile, b1)
    output = tf.nn.xw_plus_b(w1_output, w2_tile, b2, name="output")

    return output
