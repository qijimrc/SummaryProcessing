import tensorflow as tf
import six
from utils import expert_utils
from utils import devices
import collections
import math
import time
import ipdb


class base_model():
    def __init__(self,hparams,config,vocab_size,iterator,is_training,**kwargs):

        # parameters
        self.model = hparams.model
        self.iterator = iterator
        self.hparams = hparams
        self.config = config
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.init_type = hparams.init_type
        self.init_weight = hparams.init_weight
        self.seed = hparams.seed

        # inialize
        self._build_data_parallelism()
        self.init_graph()



    def add_placeholder_op(self):
        # IO
        if self.model in ('bilstm_concat','bilstm_sum'):
            if self.is_training:
                self.inputs_fw,self.inputs_bw,self.targets,self.lengths = self.iterator.get_next()
            else:
                self.inputs_fw = tf.placeholder(dtype=tf.int32,shape=[None,None])
                self.inputs_bw = tf.placeholder(dtype=tf.int32,shape=[None,None])
                self.targets = tf.placeholder(dtype=tf.int32,shape=[None,None])
                self.lengths = tf.placeholder(dtype=tf.int32,shape=[None])
            # inputs features
            self.features = {'inputs_fw':self.inputs_fw,
                             'inputs_bw':self.inputs_bw,
                             'lengths':self.lengths}
        elif self.model == 'cbow':
            if self.is_training:
                self.inputs,self.targets = self.iterator.get_next()
            else:
                self.inputs = tf.placeholder(dtype=tf.int32,shape=[None,None])
                self.targets = tf.placeholder(dtype=tf.int32,shape=[None])
            # inputs features
            self.features = {'inputs':self.inputs}
        else:
            pass

    def _build_data_parallelism(self,):
        data_parallelism = devices.data_parallelism(
            daisy_chain_variables=self.config.daisy_chain_variables,
            ps_replicas=self.config.ps_replicas,
            ps_job=self.config.ps_job,
            ps_gpu=self.config.ps_gpu,
            schedule=self.config.schedule,
            sync=self.config.sync,
            worker_gpu=self.config.num_gpus,
            worker_replicas=self.config.num_async_replicas,
            worker_id=self.config.worker_id,
            gpu_order=self.config.gpu_order,
            locally_shard_to_cpu=self.config.shard_to_cpu,
            worker_job=self.config.worker_job,
            no_data_parallelism=self.config.no_data_parallelism)
        self._data_parallelism =  data_parallelism
        self._num_datashards = self._data_parallelism.n


    def add_shard_inputs_op(self):
        sharded_features = dict()
        for k, v in sorted(six.iteritems(self.features)):
            sharded_features[k] = self._data_parallelism(
                tf.identity, tf.split(v, self._num_datashards, 0))

        datashard_features = []
        for d in range(self._num_datashards):
            f = {k: v[d] for k, v in six.iteritems(sharded_features)}
            datashard_features.append(f)
        self.datashard_features = datashard_features

    def body(self, features):
        raise NotImplementedError("Abstract Method")


    def model_fn(self, features):
        #with tf.variable_scope(tf.get_variable_scope(), use_resource=True):
        with tf.variable_scope("body"):
            print("Building model body")
            body_out = self.body(features)
        return body_out

    def add_logits_op(self):
        #dp = self._data_parallelism
        #sharded_logits = dp(self.model_fn, self.datashard_features)
        #logits = tf.concat(sharded_logits, 0)
        #self.logits = logits

        body_out = self.body(self.features)
        self.logits = body_out


    def add_pred_op(self):
        self.pred = tf.cast(tf.argmax(self.logits,-1),tf.int32)

    def add_loss_op(self):
        with tf.name_scope('loss'):
            self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.targets,logits=self.logits)
            batch_size = tf.shape(self.targets)[0]
            if 'lengths' in self.features.keys():
                max_len = tf.shape(self.targets)[1]
                mask_weights=tf.sequence_mask(
                    self.features['lengths'], max_len, dtype=tf.float32)
                self.crossent = self.crossent*mask_weights
            self.loss=tf.reduce_sum(self.crossent)/tf.to_float(batch_size)
            tf.summary.scalar('loss',self.loss)

    def add_train_op(self):
        with tf.variable_scope('train_op'):


            self.tvars = tf.trainable_variables()
            # calculate gradients
            self.grads = tf.gradients(self.loss, self.tvars)
            if self.hparams.clip_grad != None:
                self.grads, _ =tf.clip_by_global_norm(self.grads,
                                                      self.hparams.clip_grad)
            #optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
            #optimizer = tf.train.AdagradOptimizer(self.hparams.learning_rate)
            #self.train_op = optimizer.apply_gradients(zip(self.grads,self.tvars))

            optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def add_init_op(self):
        if self.init_type == 'uniform':
            assert self.init_weight
            self.initializer = tf.random_uniform_initializer(-self.init_weight,
                                                             self.init_weight,
                                                             seed=self.seed)
        elif init_type == "glorot_normal":
            self.initializer = tf.keras.initializers.glorot_normal(seed=self.seed)
        elif init_type == "glorot_uniform":
            return tf.keras.initializers.glorot_uniform(seed=self.seed)
        else:
            raise ValueError("Unknown init_op %s" % init_op)
        tf.get_variable_scope().set_initializer(self.initializer)

        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
        self.summary_merged = tf.summary.merge_all()

    def init_graph(self,):
        self.add_placeholder_op()
        self.add_shard_inputs_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()

    def run_epoch(self, session, train_op=None):
        if not train_op:
            train_op = tf.no_op()
        try:
            session.run(self.iterator.initializer)
            i = 1
            begin = time.time()
            average_loss = 0
            while True:
                loss, _ ,projection= session.run([self.loss,
                                                  train_op,
                                                  self.logits])
                average_loss += loss
                if i % 500 == 0:
                    ts = time.time()-begin
                    begin = time.time()
                    print('step [%d] batch-time [%f] loss [%f]'%(i, ts, loss))
                i += 1
        except:
            session.run(self.iterator.initializer)
            print('An epoch completed.')
            average_loss /= (i-1)
            return average_loss


    def predict(self, session, feed_dict):
        feed_dict = {self.inputs_fw:feed_dict['inputs_fw'],
                     self.inputs_bw:feed_dict['inputs_bw'],
                     self.lengths:feed_dict['lengths']}
        pred = session.run(self.pred, feed_dict=feed_dict)
        return pred
