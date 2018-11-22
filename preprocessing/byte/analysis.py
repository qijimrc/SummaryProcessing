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

def traverse(all_files,stats,cache,worker_cpu):
    # task
    def pro_lines(src_lines,cache,stats,lock,data_type='train'):

        cache = cache + '.' + data_type
        for item in src_lines:
            item = json.loads(item)
            content = item.get('content')
            title = item.get('title')

            # analysis
            # tokens
            content_tokens = nltk.tokenize.word_tokenize(content)
            title_tokens = nltk.tokenize.word_tokenize(title)
            # sents
            content_sents = nltk.tokenize.sent_tokenize(content)
            title_sents = nltk.tokenize.sent_tokenize(title)
            # chars
            content_chars = [c for c in content]
            title_chars = [c for c in title]

            stats['co_num_doc_words'].append(len(content_tokens))
            stats['ti_num_doc_words'].append(len(title_tokens))

            stats['co_num_doc_sents'].append(len(content_sents))
            stats['ti_num_doc_sents'].append(len(title_sents))

            stats['co_num_doc_chars'].append(len(content_chars))
            stats['ti_num_doc_chars'].append(len(title_chars))

            stats['co_num_sent_words'].extend([len(nltk.tokenize.word_tokenize(sent)) for sent in content_sents])
            stats['ti_num_sent_words'].extend([len(nltk.tokenize.word_tokenize(sent)) for sent in title_sents])

            if 'words_freq' not in stats.keys():
                stats['co_words_freq'].update(collections.Counter(content_tokens))
                stats['ti_words_freq'].update(collections.Counter(title_tokens))
            else:
                stats['co_words_freq'].update(collections.Counter(content_tokens))
                stats['ti_words_freq'].update(collections.Counter(title_tokens))

            if 'chars_freq' not in stats.keys():
                stats['co_chars_freq'].update(collections.Counter(content_chars))
                stats['ti_chars_freq'].update(collections.Counter(title_chars))
            else:
                stats['co_chars_freq'].update(collections.Counter(content_chars))
                stats['ti_chars_freq'].update(collections.Counter(title_chars))

        lock.acquire()
        if not os.path.exists(cache):
            with open(cache,'wb') as f:
                pickle.dump(stats,f)
        else:
            # update
            with open(cache,'rb') as f:
                old_result = pickle.load(f)
            for k in stats.keys():
                if 'num' in k:
                    old_result[k].extend(stats[k])
                else:
                    old_result[k].update(stats[k])
            with open(cache,'wb') as f:
                pickle.dump(stats,f)
        lock.release()

    # pro
    lock = Lock()
    start_time = time.time()
    cpus = worker_cpu if worker_cpu != None else cpu_count()

    # train data
    for f_i,(file_name,file_size) in enumerate(all_files['train']):
        with open(file_name,'r') as f:
            # segment
            piece = math.ceil(file_size / float(cpus))
            print('start to pro file [%d] lines, liness count:[%d], cpu count:[%d]'%(f_i,file_size,cpus))
            # run multi-process
            process_list = []
            for i in range(int(cpus)):
                lines = list(itertools.islice(f,int(piece)))
                args=(lines,cache,stats,lock)
                p = Process(target=pro_lines, args=args)
                process_list.append(p)
            start = time.time()
            for p in process_list:
                p.start()
            for p in process_list:
                p.join()
            del process_list
    print('train data completed. time conseuming:[%.4f]s' % (time.time()-start_time))

    with open(cache+'.train', 'rb') as f:
        result_stats = pickle.load(f)
    os.remove(cache+'.train')
    return result_stats


def load_files(src_dir):
    all_files = collections.defaultdict(list)
    for root,dir,files in os.walk(src_dir):
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

def save_analysis(stats, result_dir):
    # save
    for k,v in stats.items():
        if 'num' in k:
            # save
            with open(os.path.join(result_dir,k+'.txt'),'w') as f:
                sorted_v = sorted(v,reverse=True)
                f.write(' '.join(map(str,sorted_v)))
        else:
            # save
            with open(os.path.join(result_dir,k+'.txt'),'w') as f:
                sorted_items = sorted(v.items(),key=lambda v:v[1])
                for word,freq in sorted_items:
                    f.write(word+' '+str(freq)+'\n')

def show_analysis(result_dir):
    for item in os.listdir(result_dir):
        with open(os.path.join(result_dir,item),'r') as f:
            if 'num' in item:
                data = f.readlines()
                if data:
                    data = data.pop()
                    data = [int(d) for d in data.strip().split()]
                    data = np.array(data)
                    print(item+', max:[%.4f], min:[%.4f],mean:[%.4f]' %(data.max(),
                                                                        data.min(),
                                                                        data.mean()))
            else:
                contents = []
                freqs = []
                for line in f:
                    try:
                        line_list = line.strip().split()
                        content,freq = line_list[0],line_list[1]
                        contents.append(content)
                        freqs.append(int(freq))
                    except:
                        pass
                freqs = np.array(freqs)
                print(item+', max:[%.4f], min:[%.4f],mean:[%.4f]' %(freqs.max(),
                                                                    freqs.min(),
                                                                    freqs.min()))

def main(params):
   all_files = load_files(params.src_dir)
   cache = params.cache
   result = params.result
   worker_cpu = params.cpu_count
   # define stats
   stats = {'co_num_doc_words':[],
            'co_num_doc_sents':[],
            'co_num_doc_chars':[],
            'co_num_sent_words':[],
            'co_num_sent_chars':[],
            'co_words_freq':collections.Counter(),
            'co_chars_freq':collections.Counter(),
            'ti_num_doc_words':[],
            'ti_num_doc_sents':[],
            'ti_num_doc_chars':[],
            'ti_num_sent_words':[],
            'ti_num_sent_chars':[],
            'ti_words_freq':collections.Counter(),
            'ti_chars_freq':collections.Counter()
           }
   ## get result
   #result_stats = traverse(all_files,stats,cache,worker_cpu)
   ## show and save
   #save_analysis(result_stats,result)
   show_analysis(result)



if __name__=='__main__':
    # define the hyperparameters
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--src_dir',default='../../../DATA/byte/raw/')
    parser.add_argument('--cache',default='../../cache/analysis.cache')
    parser.add_argument('--result',default='../../results/analysis/')
    parser.add_argument('--cpu_count',type=int,default=None)
    # get all params
    params = parser.parse_args()

    main(params)
