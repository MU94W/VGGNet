import tensorflow as tf
from TFCommon.Model import Model
from TFCommon.metrics import binary_accuracy

config_16 = {"input": (224, 224, 3),
        "conv_layers": [2, 2, 3, 3, 3],
        "conv_channels": [64, 128, 256, 512, 512],
        "dense_units": [4096, 4096, 1000],
        "lambda_l1": 0.,
        "lambda_l2": 0.,}

def flatten_tensor(ts):
    """
    Args:
        ts: shape +=> (batch_size, d1, d2, ..., dn), d1~dn must be defined.
    """
    batch_size = tf.shape(ts)[0]
    assert ts.shape[1:].is_fully_defined(), "[!] tensor's d1~dn is not fully defined."
    dim_shape = ts.shape.as_list()[1:]
    flat_dim = 1
    for dim in dim_shape:
        flat_dim *= dim
    return tf.reshape(ts, (batch_size, flat_dim))


class VGG(Model):
    """
    """

    def __init__(self, config=config_16, name="VGG16"):
        """
        """
        self.__config = config
        self.__name = name

    @property
    def config(self):
        return self.__config

    @property
    def name(self):
        return self.__name

    @property
    def loss(self):
        return self.__loss

    @property
    def metric(self):
        return self.__metric

    @property
    def dense_hid_lst(self):
        return self.__dense_hid_lst

    @property
    def update(self):
        return self.__upd

    @property
    def input(self):
        return self.__inp

    @property
    def output(self):
        return self.__out

    def build(self, scope=None):
        """
        """
        with tf.variable_scope(scope or "model"):
            with tf.variable_scope("data"):
                self.__inp = tf.placeholder(name="inp", \
                        shape=(None,) + self.config.get("input"), dtype=tf.float32)
                self.__out = tf.placeholder(name="out", \
                        shape=(None, 1), dtype=tf.float32)
            loss = self.__build_forward(self.__inp, self.__out)
            upd = self.__build_backprop(loss)
            return upd


    def __build_forward(self, inp, out, scope=None):
        """
        """
        with tf.variable_scope(scope or "forward"):
            with tf.variable_scope("conv"):
                layers = self.config.get("conv_layers")
                channels = self.config.get("conv_channels")
                assert len(layers) == len(channels), "[!] CONFIG ERROR."
                last_out = inp
                for lay_idx in range(len(layers)):
                    with tf.variable_scope("lay_%d" % lay_idx):
                        out_channels = channels[lay_idx]
                        for inner_idx in range(layers[lay_idx]):
                            with tf.variable_scope("inner_%d" % inner_idx):
                                in_channels = last_out.shape[-1].value
                                this_filter = tf.get_variable(name="filter", \
                                        shape=(3, 3, in_channels, out_channels))
                                this_out = tf.nn.relu(tf.nn.conv2d(last_out, this_filter, [1, 1, 1, 1], "SAME"))
                                last_out = this_out
                        last_out = tf.nn.max_pool(last_out, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

            with tf.variable_scope("dense"):
                last_out = flatten_tensor(last_out)
                dense_hid_lst = []
                layers = self.config.get("dense_units")[:-1]    # drop the final classification layer
                for lay_idx in range(len(layers)):
                    with tf.variable_scope("lay_%d" % lay_idx):
                        units = layers[lay_idx]
                        this_out = tf.layers.dense(last_out, units, tf.nn.relu)
                        dense_hid_lst.append(this_out)
                        last_out = this_out
                self.__dense_hid_lst = dense_hid_lst

            with tf.variable_scope("classification"):
                logits = tf.layers.dense(last_out, 1, None)

            self.__predict = tf.sigmoid(logits)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=out, logits=logits)
            metric = binary_accuracy(out, self.__predict)
            train_vars = tf.trainable_variables()
            lambda_l1 = self.config.get("lambda_l1")
            lambda_l2 = self.config.get("lambda_l2")
            l1_reg = tf.contrib.layers.l1_regularizer(lambda_l1)
            l2_reg = tf.contrib.layers.l2_regularizer(lambda_l2)
            l1_loss = tf.contrib.layers.apply_regularization(l1_reg, train_vars)
            l2_loss = tf.contrib.layers.apply_regularization(l2_reg, train_vars)
            loss += l1_loss + l2_loss
            self.__loss = loss
            self.__metric = metric
            return loss

    def __build_backprop(self, loss, scope=None):
        """
        """
        with tf.variable_scope(scope or "backprop"):
            l_rate = tf.Variable(0.01, name="learning_rate", trainable=False)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            opt = tf.train.AdamOptimizer()
            #opt = tf.train.GradientDescentOptimizer(l_rate)
            upd = opt.minimize(loss, global_step=global_step)
            
            self.__upd = upd
            return upd


