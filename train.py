from data import AttnDataSet
from network import attention_network   # ,network
from tensorflow.contrib import seq2seq
from config import *
from collections import Counter
import tensorflow as tf
import numpy as np
import random
import os


def train():
    dataset = AttnDataSet(batch_size, max_length, vocab_size)

    # 各种变量
    X_encoder = tf.placeholder(tf.float32, [batch_size, max_length, vector_size])
    X_decoder = tf.placeholder(tf.float32, [batch_size, max_length, vector_size])
    encoder_length = tf.placeholder(tf.int32, [batch_size])
    decoder_length = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size, max_length])
    z_encoder = tf.placeholder(tf.int32, [batch_size, max_length])
    z_decoder = tf.placeholder(tf.int32, [batch_size, max_length])
    dropout_prob = tf.placeholder(tf.float32)
    mask = tf.placeholder(tf.float32, [batch_size, max_length])
    # 注意力层
    output, decoder_state = attention_network(X_encoder, X_decoder, encoder_length, z_encoder,
                                                      z_decoder, dropout_prob)
    alignments = decoder_state[3]
    alignment_history = decoder_state[4]

    # Loss Function
    seq_loss = seq2seq.sequence_loss(output, y, mask)

    # 变量分类
    attention_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="attention_scope")   # attention变量

    trainableVariables = tf.trainable_variables(scope="attention_scope")                # 只对attention_scope中的变量进行更新
    grads, a = tf.clip_by_global_norm(tf.gradients(seq_loss, trainableVariables), 5)    # 限制梯度上限, 防止梯度爆炸

    # 迭代次数
    globalStep = tf.Variable(0, trainable=False)
    addGlobalStep = globalStep.assign_add(1)
    # 迭代次数增加, 学习率自动衰减
    learningRate = tf.train.exponential_decay(learningRateBase, global_step=globalStep,
                                              decay_steps=learningRateDecayStep, decay_rate=learningRateDecayRate)

    optimizer = tf.train.AdamOptimizer(learningRate)  # 优化器
    train_op = optimizer.apply_gradients(zip(grads, trainableVariables))  # 梯度下降

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # 训练Attention相关参数
        saver = tf.train.Saver(max_to_keep=10, var_list=attention_vars)
        print("训练参数列表:")
        print(attention_vars)
        total_loss = 0
        model_dir = "./model"
        print("is training")

        if not os.path.exists(model_dir):   # 检查./model路径是否存在
            os.mkdir(model_dir)             # 不存在就创建路径
            print("create the directory: %s" % model_dir)

        checkPoint = tf.train.get_checkpoint_state(model_dir)
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")

        while dataset.epoch < total_epoch:  # 迭代训练
            _X_encoder, _X_decoder, _encoder_length, _decoder_length, _y, _z_encoder, _z_decoder, _mask = dataset.get_next_batch()

            _z_encoder = np.zeros((batch_size, max_length), dtype=np.int32)     # encoder的位置编码用全0表示, 否则网络学习很容易只学习到基于位置编码的特征

            _, _loss, step = sess.run([train_op, seq_loss, globalStep],
                                    feed_dict={X_encoder: _X_encoder, X_decoder: _X_decoder, encoder_length: _encoder_length,
                                         decoder_length: _decoder_length, y: _y, z_encoder: _z_encoder,
                                         z_decoder: _z_decoder, mask: _mask, dropout_prob: 0.8})
            total_loss += _loss
            if step % 100 == 0:
                # 打印当前训练情况
                print("epoch:", dataset.epoch, "step:", step, "loss:", total_loss / 100)
                total_loss = 0

            if step % 2000 == 0:
                # 打印首词的注意力权重分布
                _logists, al, al_history = sess.run([output, alignments, alignment_history.read(0)],
                                    feed_dict={X_encoder: _X_encoder, X_decoder: _X_decoder, encoder_length: _encoder_length,
                                         decoder_length: _decoder_length, y: _y, z_encoder: _z_encoder,
                                         z_decoder: _z_decoder, mask: _mask, dropout_prob: 1.})
                for i in range(min(batch_size, 5)):
                    print("目标:", dataset.sentence_decode(_y[i]))
                    print("实际:", dataset.sentence_decode(_logists.argmax(axis=2)[i]))  # 当前batch的第一个输出结果
                    print("首词注意力:", al_history[i].tolist())
            if step % 10000 == 0:
                # 保存模型
                saver.save(sess, model_dir + '/attention_model', global_step=step)  # 保存模型
                print("saving...")
            sess.run(addGlobalStep)


if __name__ == "__main__":
    train()
