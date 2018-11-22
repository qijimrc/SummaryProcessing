import os
import argparse
import codecs
from nltk.parse.corenlp import CoreNLPParser

def make_submission(args):
    '''cd stanford-corenlp-full-2018-01-31 java -mx4g -cp "*"
    edu.stanford.nlp.pipeline.StanfordCoreNLPServer \ -preload
    tokenize,ssplit,pos,lemma,parse,depparse \ -status_port 9000 -port 9000
    -timeout 15000 '''

    stanford = CoreNLPParser()
    src = args.src
    tgt = args.tgt

    with open(src,'r') as src_f:
        for idx,line in enumerate(src_f):
            # process
            result = line.strip()
            result = list(stanford.tokenize(result))
            # save
            tgt_f_name = os.path.join(tgt,str(idx+1)+'.txt')
            with codecs.open(tgt_f_name,'w','utf-8') as tgt_f:
                tgt_f.write(' '.join(result)+'\n')




if __name__=='__main__':

    # define the hyperparameters
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--src',default='../../OpenNMT-py/results/byte.out')
    parser.add_argument('--tgt',default='../../OpenNMT-py/results/result/')
    args = parser.parse_args()

    make_submission(args)
