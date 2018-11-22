#coding=utf-8
import tensorflow as tf
from multiprocessing import cpu_count, Process, Lock
from preprocessing import preprocessing_slices as psl
from collections import Counter
import config
import numpy as np
import pickle
import copy
import os
import re
import math
import time
import ipdb



class BiLSTMGenerator():
    def __init__(self, files_dir, params):

        # parameters
        self.files = self._load_files(files_dir)
        self.vocab, self.vocab_size = self._load_vocab(params.vocab_file)
        self.cover_window = params.cover_window
        self.batch_size = params.batch_size
        self.seq_len = params.seq_len

        # iterator
        self.iterator = self.get_iterator()


    def _load_files(self,files_dir):
        files = [os.path.join(files_dir,f) for f in os.listdir(files_dir)]
        return files

    def _load_vocab(self,vocab_file):
        if not os.path.exists(vocab_file):
            raise BaseException('Vocab file not found.')
        with open(vocab_file,'r') as f:
            items = f.readlines()
        items = list(map(lambda line:(line.split()[0],int(line.split()[1])),
                         items))
        return dict(items), len(items)


    def get_iterator(self):
        batch_size = self.batch_size
        list_file = self.files
        k = self.cover_window
        def batching_func(x):
            return x.padded_batch(batch_size,padded_shapes=
                                  (tf.TensorShape([None]),tf.TensorShape([None]),
                                   tf.TensorShape([None]),tf.TensorShape([])),
                                  padding_values=(0,0,0,0))
        dataset = tf.data.Dataset.from_tensor_slices(list_file)
        dataset = dataset.flat_map(lambda filename:(tf.data.TextLineDataset(filename)))
        dataset = dataset.map(lambda x:tf.to_int32(tf.string_to_number((tf.string_split([x]).values))))
        dataset =dataset.map(lambda x:(x[:-2*k],x[2*k:],x[k:-k],tf.size(x[:-k])))
        batched_dataset = batching_func(dataset.shuffle(buffer_size=100*batch_size))
        iterator = batched_dataset.make_initializable_iterator()
        return iterator
