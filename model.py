import tensorflow as tf, numpy as np
from TFCommon.Model import Model
from TFCommon.metrics import binary_accuracy

config_16 = {"input": (224, 224, 3),
        "conv_layers": [2, 2, 3, 3, 3],
        "conv_channels": [64, 128, 256, 512, 512],
        "dense_units": [4096, 4096, 1000],  # the last layer's units actually is 1.
        "hash_codes": 48,
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


class modifiedVGG(Model):
    """
    """

    def __init__(self, sess, use_hash=True, config=config_16, save_path="save", opt="adam", start_lr=0.05, name="modifiedVGG16"):
        """
        """
        self.__sess = sess
        self.__use_hash = use_hash
        self.__config = config
        self.__opt = opt
        self.__start_lr = start_lr
        self.__log = tf.summary.FileWriter("log", sess.graph)
        self.__save_path = save_path
        self.__name = name

    @property
    def use_hash(self):
        return self.__use_hash

    @property
    def log(self):
        return self.__log

    @property
    def save_path(self):
        return self.__save_path

    @property
    def sess(self):
        return self.__sess

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

    @property
    def global_step(self):
        return self.__global_step

    @property
    def train_summary(self):
        return self.__train_summary

    @property
    def dev_summary(self):
        return self.__dev_summary

    def __add_summary(self, suffix):
        sums = []
        sums.append(tf.summary.scalar("%s/loss" % suffix, self.loss))
        sums.append(tf.summary.scalar("%s/acc" % suffix, self.metric))
        return tf.summary.merge(sums)

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
            self.__train_summary = self.__add_summary("train")
            self.__dev_summary = self.__add_summary("dev")
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            return upd

    def fit(self, train_data, dev_data, batch_size=32, epochs=20, summary_step=10, save_step=100):
        """
        """
        train_inp = train_data.get("input")
        train_out = train_data.get("output")
        dev_inp = dev_data.get("input")
        dev_out = dev_data.get("output")
        samples = train_out.shape[0]
        for epoch in range(epochs):
            print("Epoch %d / %d" % (epoch+1, epochs))
            perm = np.random.permutation(samples)
            start = 0
            end = start + batch_size
            while start < samples:
                global_step = self.sess.run(self.global_step)
                perm_index = perm[start:end]
                feed_dict = {self.input: train_inp[perm_index],\
                        self.output: train_out[perm_index]}
                if (global_step+1) % summary_step != 0:
                    self.train(min(end, samples), samples, feed_dict)
                else:
                    self.train(min(end, samples), samples, feed_dict, True)
                    self.evaluate({self.input: dev_inp, self.output: dev_out})
                if (global_step+1) % save_step == 0:
                    print("Save model ...")
                    self.save(self.save_path, global_step)
                start = end
                end = start + batch_size
            print("")
        global_step = self.sess.run(self.global_step)
        print("Fit finish. Save model ...")
        self.save(self.save_path, global_step)


    def train(self, used_samples_cnt, total_samples, feed_dict, eval_summary=False):
        """
        """
        if not eval_summary:
            loss, acc, _ = self.sess.run([self.loss, self.metric, self.update], feed_dict)
        else:
            loss, acc, summary, global_step, _ = self.sess.run([self.loss, self.metric, self.train_summary, \
                    self.global_step, self.update], feed_dict)
            self.log.add_summary(summary, global_step)
        ### show some useful info.
        console_log = "\r[%d / %d]:\tloss: %f; accuracy: %f" % (used_samples_cnt, total_samples, loss, acc)
        print(console_log, end="")

    def evaluate(self, feed_dict):
        loss, acc, summary, global_step = self.sess.run([self.loss, self.metric, self.dev_summary, \
                self.global_step], feed_dict)
        self.log.add_summary(summary, global_step)
        console_log = "\n[dev]:\tloss: %f; accuracy: %f" % (loss, acc)
        print(console_log)
        

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

            if self.use_hash:
                with tf.variable_scope("hash_layer"):
                    units = self.config.get("hash_codes")
                    this_out = tf.layers.dense(last_out, units, tf.sigmoid)
                    dense_hid_lst.append(this_out)
                    last_out = this_out

            self.__dense_hid_lst = dense_hid_lst

            with tf.variable_scope("classification"):
                logits = tf.layers.dense(last_out, 1, None)

            self.__predict = tf.sigmoid(logits)

            loss = tf.losses.sigmoid_cross_entropy(out, logits)
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
            l_rate = tf.Variable(self.__start_lr, name="learning_rate", trainable=False)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.__opt == "adam":
                opt = tf.train.AdamOptimizer()
            elif self.__opt == "sgd":
                opt = tf.train.GradientDescentOptimizer(l_rate)
            upd = opt.minimize(loss, global_step=global_step)

            self.__global_step = global_step
            self.__upd = upd
            return upd


