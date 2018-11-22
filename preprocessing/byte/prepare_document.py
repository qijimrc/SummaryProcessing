#coding=utf-8
from multiprocessing import cpu_count, Process, Lock
from collections import OrderedDict
from collections import Counter
import pickle
import time
import math
import os
import sys
import argparse
import re
import numpy as np
import ipdb

CONSTANT = OrderedDict([
    ('PAD','<pad>'),
    ('UNK','<unk>'),
    ('BEGIN','<s>'),
    ('END','</s>'),
    ('NUM','<num>'),
])

re_rpl_space = [(r'\n',' '),(r'\t',' ')]
#re_need_chinese = u'.*?[\u4e00-\u9fa5]{2}.*'
#line_delimiters = ['，','。','！','？']

class PrepareTrain():
    def __init__(self, params):
        # parameters
        self.vocab_threshold = params.vocab_threshold
        self.vocab_file = params.vocab_file
        self.cache_dir = params.cache_dir
        self.corpus_dir = params.corpus_dir
        self.cpu_count = params.cpu_count
        self.train_dir = params.train_dir
        self.shuffle = params.shuffle
        self.seq_len = params.seq_len

        # build vocabulary
        self.vocabulary,self.reverse_vocabulary,self.vocab_size=self.build_vocabulary()
        # generate dadaset
        self.generate_dataset()

    def build_vocabulary(self,):
        def count_files_word(files, cache_counter, lock):
            # word frequency results
            counter = Counter()
            for fl in files:
                with open(fl, 'r') as f:
                    text = f.read()
                    tokens = self.tokenizer(text)
                    counter.update(tokens)
            # update cache counter
            lock.acquire()
            if not os.path.exists(cache_counter):
                with open(cache_counter, 'wb') as f:
                    pickle.dump(file=f, obj=counter)
            else:
                with open(cache_counter, 'rb') as f:
                    cc = pickle.load(f)
                    cc.update(counter)
                with open(cache_counter, 'wb') as f:
                    pickle.dump(file=f, obj=cc)
            lock.release()

        if not os.path.exists(self.vocab_file): # create new vocab
            # Rules for vocab map
            lock = Lock()

            #worker cpu
            cpus = self.cpu_count if self.cpu_count != None else cpu_count()

            # search for all files
            all_files = []
            for items in os.walk(self.corpus_dir):
                if len(items[2]) > 0:
                    f_dir = os.path.abspath(items[0])
                    for f_name in items[2]:
                        all_files.append(os.path.join(f_dir,f_name))
            # files segment
            part_n = math.ceil(len(all_files) / float(cpus))
            seg_ids = np.arange(0, len(all_files), part_n)
            if seg_ids[-1] < len(all_files):
                seg_ids = list(seg_ids)
                seg_ids.append(len(all_files))

            # run multi-process
            cache_dir = os.path.abspath(self.cache_dir)
            cache_file = os.path.join(cache_dir,'counter.cache')
            process_list = []
            for i in range(1, len(seg_ids)):
                files = all_files[seg_ids[i - 1]:seg_ids[i]]
                p = Process(target=count_files_word, args=(files, cache_file, lock))
                process_list.append(p)
            start = time.time()
            for p in process_list:
                p.start()
            for p in process_list:
                p.join()
            end = time.time()
            print('build vocab time consuming:[%.4f]' % (end - start))

            # analysis results
            with open(cache_file, 'rb') as f:
                final_counter = pickle.load(f)
            os.system('rm ' + cache_file)

            final_counter = sorted(final_counter.items(), key=lambda v: v[1], reverse=True)
            # save
            with open(self.vocab_file,'w') as f:
                for idx,const in enumerate(CONSTANT.values()):
                    f.write(const + ' ' + str(idx) + '\n')
                for idx,(word,count) in enumerate(final_counter):
                    if word in CONSTANT.values(): continue
                    if count <= self.vocab_threshold: break
                    f.write(word+ ' ' + str(idx+len(CONSTANT))+'\n')
        # build vocab
        dictionary = dict()
        with open(self.vocab_file, 'r') as f:
            for line in f:
                line_text = line.strip()
                word, idx = line_text.split(' ')
                dictionary[word] = int(idx)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return dictionary, reverse_dictionary, len(dictionary)


    def generate_dataset(self,):
        # task
        def pro_files(src_files,train_dir,val_dir,test_dir,seq_len,
                      begin_count,end_count,begin,end,pad,
                      subfix,train_prob,val_prob,test_prob):

            tgt_train_f =open(os.path.join(train_dir,'train_'+subfix+'.txt'),'w')
            tgt_val_f = open(os.path.join(val_dir,'val'+subfix+'.txt'),'w')
            tgt_test_f = open(os.path.join(test_dir,'test'+subfix+'.txt'),'w')
            for src_file in src_files:
                with open(src_file,'r') as f:
                    for line in f:
                        line_text = self.substitute_sequence(line,re_rpl_space)
                        line_list = []
                        self.split_sequence_keep_delimiters(line_text,
                                                            line_delimiters,
                                                            line_list)
                        for text in line_list:
                            if re.match(re_need_chinese,text):
                                tokens = self.tokenizer(text)
                                if len(tokens) <= seq_len-begin_count-end_count:
                                    # convert to id
                                    UNK = CONSTANT['UNK']
                                    tokens = [self.vocabulary[c] if c in
                                              self.vocabulary.keys() else
                                              self.vocabulary[UNK] for c in tokens]
                                    tokens = list(map(str, tokens))
                                    # pad
                                    tokens=[str(begin)]*begin_count+tokens+[str(end)]*end_count
                                    tokens += [str(pad)]*(seq_len-len(tokens))
                                    # Whether to add test file
                                    prob = np.random.rand()
                                    if prob <= train_prob:  # add train source
                                        save_f = tgt_train_f
                                    elif prob <= train_prob + test_prob:
                                        save_f = tgt_test_f
                                    else:
                                        save_f = tgt_val_f
                                    # write
                                    save_f.write(' '.join(tokens)+'\n')
            tgt_train_f.close()
            tgt_val_f.close()
            tgt_test_f.close()

        # search for all files
        all_files = []
        for items in os.walk(self.corpus_dir):
            if len(items[2]) > 0:
                f_dir = os.path.abspath(items[0])
                for f_name in items[2]:
                    all_files.append(os.path.join(f_dir,f_name))
        # shuffle
        if self.shuffle:
            np.random.shuffle(all_files)
        cpus = self.cpu_count if self.cpu_count != None else cpu_count()
        # pro
        print('start to pro all files, files count:[%d], cpu count:[%d]'%(len(all_files),cpus))

        # files segment
        part_n =  math.ceil(len(all_files) / float(cpus))
        seg_ids = np.arange(0,len(all_files)-1,part_n)
        if seg_ids[-1] < len(all_files):
            seg_ids = list(seg_ids)
            seg_ids.append(len(all_files))

        # run multi-process
        process_list = []
        begin = self.vocabulary[CONSTANT['BEGIN']]
        end = self.vocabulary[CONSTANT['END']]
        pad = self.vocabulary[CONSTANT['PAD']]
        for i in range(1, len(seg_ids)):
            files = all_files[seg_ids[i - 1]:seg_ids[i]]
            args=(files,self.train_dir,self.val_dir,self.test_dir,self.seq_len,
                  self.begin_count,self.end_count,begin,end,pad,
                  str(i),self.train_prob,self.val_prob,self.test_prob)
            p = Process(target=pro_files, args=args)
            process_list.append(p)
        start = time.time()
        for p in process_list:
            p.start()
        for p in process_list:
            p.join()
        end = time.time()
        print('time conseuming:[%.4f]' % (end - start))

    def substitute_sequence(self,line_text,rules):
        # use python regularization
        for pattern,rpl in rules: # replace
            line_text=re.sub(pattern,rpl,line_text)
        return line_text


    # Recursive split sequence using all delimiters
    def split_sequence(self,line_text, dls, results):
        if len(dls) >0:
            result_list = re.split(dls[0], line_text)
            if len(dls) == 1:
                for ll in result_list:
                    if ll != '':
                        results.append(ll)
            else:
                for text in result_list:
                    self.split_sequence(text, dls[1:],results )

    # Recursive split sequence keep delimiter
    def split_sequence_keep_delimiters(self,line_text, dls, results):
        if len(dls) >0:
            result_list = re.split(dls[0], line_text)
            if len(result_list) > 1:
                result_list[:-1] = [rl+dls[0] for rl in result_list[:-1]]
            if len(dls) == 1:
                for ll in result_list:
                    if ll != '':
                        results.append(ll)
            else:
                for text in result_list:
                    self.split_sequence_keep_delimiters(text, dls[1:],results)

    def tokenizer(self, text):
        re_rpl_space = [(r'\s',''),(r'\n',''),(r'\t','')]
        re_rpl_num = [(r'\d+','0')]
        re_rpl_eng = [(r'[a-zA-Z]+','a')]

        text = self.substitute_sequence(text, re_rpl_space)
        text = self.substitute_sequence(text, re_rpl_num)
        text = self.substitute_sequence(text, re_rpl_eng)

        tokens = []
        for c in text:
            if c == '0':
                c = CONSTANT['NUM']
            elif c == 'a':
                c = CONSTANT['ENG']
            tokens.append(c)
        return tokens







if __name__=='__main__':

    # define the hyperparameters
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--cpu_count',type=int,default=None)
    parser.add_argument('--shuffle',default=True)
    parser.add_argument('--vocab_threshold',type=int,default=2)
    parser.add_argument('--seq_len',type=int,default=60)#final
    parser.add_argument('--vocab_file',default='../data1/vocab.txt')
    parser.add_argument('--cache_dir',default='../cache/')
    parser.add_argument('--corpus_dir',default='../../DATA/公文_武汉')
    parser.add_argument('--train_dir',default='../data1/train/')
    # get all params
    params = parser.parse_args()

    PrePro = PrepareTrain(params)
