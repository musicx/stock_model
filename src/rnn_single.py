import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import QUANTAXIS as qa

HIDDEN_SIZE = 30                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。

pvars = ['open', 'close', 'high', 'low']

def generate_data(stock):
    try:
        candles = qa.QA_fetch_stock_day_adv(stock, start='2014-01-01', end='2018-05-02')
        cdata = candles.data.reset_index(level=1, drop=True).loc[:, pvars]
        # mid = []
        data = []
        label = []
        for x in range(cdata.shape[0]-TIMESTEPS+1):
            # mid.append(cdata.iloc[x: x+TIMESTEPS, :])
            data.append(np.asarray(cdata.iloc[x: x+TIMESTEPS, :] / cdata.ix[x+TIMESTEPS-1, 'close'] - 1))
            label.append(cdata.ix[x+TIMESTEPS, 'close'] / cdata.ix[x+TIMESTEPS-1, 'close'] - 1)
    except:
        pass
    return np.asarray(data, dtype=np.float32), np.asarray(label, dtype=np.float32)

data, label = generate_data('000001')

train_X = data[:800]
train_y = label[:800]
test_X = data[800:]
test_y = label[800:]

print(len(data))


def lstm_model(X, y, is_training):
    # 使用多层的LSTM结构。
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    # predictions.set_shape([None, 1])
    # y.set_shape([None, 1])
    print(tf.shape(predictions))
    print(y.shape)

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None

    # 计算损失函数。
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return predictions, loss, train_op


def run_eval(sess, test_X, test_y):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(len(test_X)):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算rmse作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 对预测的sin函数曲线进行绘图。
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_close')
    plt.legend()
    plt.show()


# 将训练数据以数据集的方式提供给计算图。
ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
X, y = ds.make_one_shot_iterator().get_next()

# 定义模型，得到预测结果、损失函数，和训练操作。
with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(X, y, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 测试在训练之前的模型效果。
    print("Evaluate model before training.")
    run_eval(sess, test_X, test_y)

    # 训练模型。
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))

    # 使用训练好的模型对测试数据进行预测。
    print("Evaluate model after training.")
    run_eval(sess, test_X, test_y)