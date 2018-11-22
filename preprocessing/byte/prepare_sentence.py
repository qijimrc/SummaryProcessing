#coding=utf-8
from multiprocessing import cpu_count, Process, Lock
import collections
import numpy as np
import itertools
import pickle
import json
import time
import math
import os
import sys
import argparse
import nltk
import re

CONSTANT = collections.OrderedDict([
    ('PAD','<pad>'),
    ('UNK','<unk>'),
    ('BEGIN','<s>'),
    ('END','</s>')
])

ENTITIES = collections.OrderedDict([
    ('\￨','/ent1'),
    #('\s+\d+\s+','/num')
])


class PrepareSentence():
    def __init__(self, params):
        # parameters
        self.use_vocab = params.use_vocab
        self.vocab_min_freq = params.vocab_min_freq
        self.max_word_len = params.max_word_len
        self.vocab_file = params.vocab_file
        self.cache_dir = params.cache_dir
        self.pad_bd = params.pad_bd
        self.src_dir = params.src_dir
        self.tgt_dir = params.tgt_dir
        self.cpu_count = params.cpu_count
        self.shuffle = params.shuffle

        # load all files
        self.all_files = self.load_files()
        # build vocabulary
        if self.use_vocab:
            self.vocabulary,self.reverse_vocabulary,self.vocab_size=self.build_vocabulary()
        # generate dadaset
        self.gen_train_data()
        self.gen_val_data()

    def load_files(self,):
        all_files = collections.defaultdict(list)
        for root,dir,files in os.walk(self.src_dir):
            if 'train' in root:
                for file in files:
                    file_name = os.path.join(root,file)
                    file_size = 0
                    with open(file_name,'r') as f:
                        for _ in f:
                            file_size += 1
                    all_files['train'].append((file_name,file_size))
            elif 'val' in root:
                if 'val_results' in root:
                    for file in files:
                        all_files['val_titles'].append(os.path.join(root,file))
                else:
                    for file in files:
                        all_files['val_contents'] = os.path.join(root,file)

        return all_files

    def build_vocabulary(self,):
        def count_lines_word(lines, cache_counter, lock):
            # word frequency results
            counter = collections.Counter()
            for line in lines:
                item = json.loads(line)
                content = item.get('content')
                title = item.get('title')
                content_tokens = self.tokenizer(content)
                title_tokens = self.tokenizer(title)
                tokens = content_tokens+title_tokens
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
            print('Vocabulary not found, start to build...')
            # Rules for vocab map
            lock = Lock()
            cache_dir = os.path.abspath(self.cache_dir)
            cache_file = os.path.join(cache_dir,'counter.cache')
            #worker cpu
            cpus = self.cpu_count if self.cpu_count != None else cpu_count()

            for file_name,file_size in self.all_files['train']:
                with open(file_name,'r') as f:
                    # segment
                    piece = math.ceil(file_size / float(cpus))
                    # run multi-process
                    process_list = []
                    for i in range(cpus):
                        lines = list(itertools.islice(f,int(piece)))
                        p = Process(target=count_lines_word, args=(lines, cache_file, lock))
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
                for i,const in enumerate(CONSTANT.values()):
                    f.write(const + ' ' + str(i) + '\n')
                for i,ent in enumerate(ENTITIES.values()):
                    f.write(ent + ' ' + str(i+j))
                for k,(word,count) in enumerate(final_counter):
                    if word in (*CONSTANT.values(), *ENTITIES.values()): continue
                    if count <= self.vocab_min_freq: break
                    f.write(word+ ' ' + str(k+j)+'\n')
        # build vocab
        dictionary = dict()
        with open(self.vocab_file, 'r') as f:
            for line in f:
                line_text = line.strip()
                word, idx = line_text.split(' ')
                dictionary[word] = int(idx)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        print('load vocabulary completed.')
        return dictionary, reverse_dictionary, len(dictionary)


    def gen_train_data(self,):
        # task
        def pro_lines(src_lines,tgt_dir,begin,end,name,lock):
            content_results = []
            title_results = []
            for item in src_lines:
                item = json.loads(item)
                content = self.tokenizer(item.get('content'))
                title = self.tokenizer(item.get('title'))
                if self.use_vocab:
                    # convert to id
                    content = [self.vocabulary[c] if c in
                              self.vocabulary.keys() else
                              self.vocabulary[CONSTANT['UNK']] for c in content]
                    content = list(map(str, content))
                    title = [self.vocabulary[c] if c in
                              self.vocabulary.keys() else
                              self.vocabulary[CONSTANT['UNK']] for c in title]
                    content = list(map(str, content))
                    title = list(map(str, title))
                if self.pad_bd:
                    title = [begin]+title+[end]
                content_results.append(' '.join(content)+'\n')
                title_results.append(' '.join(title)+'\n')

            lock.acquire()
            with open(os.path.join(tgt_dir,name+'_content'+'.txt'),'w') as f:
                f.writelines(content_results)
            with open(os.path.join(train_dir,name+'_title'+'.txt'),'w') as f:
                f.writelines(title_results)
            lock.release()

        # pro
        lock = Lock()
        start_time = time.time()
        cpus = self.cpu_count if self.cpu_count != None else cpu_count()
        train_dir = os.path.join(self.tgt_dir,'train')
        val_dir = os.path.join(self.tgt_dir,'val')

        begin = None
        end = None
        if self.pad_bd:
            begin = self.vocabulary[CONSTANT['BEGIN']] if self.use_vocab else CONSTANT['BEGIN']
            end = self.vocabulary[CONSTANT['END']] if self.use_vocab else CONSTANT['END']
        # train data
        for f_i,(file_name,file_size) in enumerate(self.all_files['train']):
            with open(file_name,'r') as f:
                # segment
                piece = math.ceil(file_size / float(cpus))
                print('start to pro file [%d] lines, liness count:[%d], cpu count:[%d]'%(f_i,file_size,cpus))
                # run multi-process
                process_list = []
                for i in range(int(cpus)):
                    lines = list(itertools.islice(f,int(piece)))
                    args=(lines,train_dir,begin,end,'train_'+str(f_i)+'_'+str(i),lock)
                    p = Process(target=pro_lines, args=args)
                    process_list.append(p)
                start = time.time()
                for p in process_list:
                    p.start()
                for p in process_list:
                    p.join()
                del process_list
        print('train data completed. time conseuming:[%.4f]s' % (time.time()-start_time))


    def gen_val_data(self,):
        # val data
        start = time.time()
        # content
        with open(self.all_files['val_contents'],'r') as src_f:
            with open(os.path.join(self.tgt_dir,'test','test_contents.txt'),'w') as tgt_f:
                for line in src_f:
                    item = json.loads(line)
                    content = self.tokenizer(item.get('content'))
                    if self.use_vocab:
                        content = [self.vocabulary[c] if c in
                                  self.vocabulary.keys() else
                                  self.vocabulary[CONSTANT['UNK']] for c in content]
                        content = map(str,content)
                    tgt_f.write(' '.join(content)+'\n')
        # title
        title_files = self.all_files['val_titles']
        title_files = sorted(title_files,key=lambda v:int(re.match('^.*?(\d+).txt$',v).group(1)))
        with open(os.path.join(self.tgt_dir,'test','test_titles.txt'),'w') as tgt_f:
            for file in title_files:
                with open(file,'r') as f:
                    title = f.readlines()[0]
                title = self.tokenizer(title)
                tgt_f.write(' '.join(title)+'\n')
        end = time.time()
        print('val data completed. time conseuming:[%.4f]s' % (end - start))

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
        # max len
        if self.max_word_len:
            re_rpl_max = [(r'[^\s]{'+str(self.max_word_len)+',}',' ')]
            text = self.substitute_sequence(text, re_rpl_max)

        # entities
        re_rpl_ents = ENTITIES.items()
        text = self.substitute_sequence(text, re_rpl_ents)

        tokens = nltk.tokenize.word_tokenize(text)
        return tokens



if __name__=='__main__':

    # define the hyperparameters
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--cpu_count',type=int,default=None)
    parser.add_argument('--shuffle',default=True)
    parser.add_argument('--use_vocab',type=bool,default=False)
    parser.add_argument('--pad_bd',type=bool,default=False)
    parser.add_argument('--vocab_min_freq',type=int,default=2)
    parser.add_argument('--max_word_len',type=int,default=None)
    parser.add_argument('--vocab_file',default='../../../DATA/byte/prepared/')
    parser.add_argument('--cache_dir',default='../cache/')
    parser.add_argument('--src_dir',default='/DATA/disk1/qiji1/byte/raw/')
    parser.add_argument('--tgt_dir',default='/DATA/disk1/qiji1/byte/prepared/')
    # get all params
    params = parser.parse_args()

    PrePro = PrepareSentence(params)
