from models.aan import AAN_Model
from models.cbow import CBOW_Model
from models.bilstm_sum import BiLSTM_Sum_Model
from models.bilstm_concat import BiLSTM_Concat_Model
from generators.bilstm_generator import BiLSTMGenerator
from generators.cbow_generator import CBOWGenerator
from utils import util
import tensorflow as tf
import os
import argparse
import time
import config
import collections
import ipdb


def train(params, runconfig):

    # build problems
    if params.model in ('bilstm_sum','bilstm_concat'):
        train_data_generator = BiLSTMGenerator(params.train_dir, params)
        val_data_generator = BiLSTMGenerator(params.val_dir, params)
    elif params.model == 'cbow':
        train_data_generator = CBOWGenerator(params.train_dir, params)
        val_data_generator = CBOWGenerator(params.val_dir, params)
    elif params.model == 'aan':
        pass
    else:
        raise BaseException('Please select a model.')
    #train_iterator = train_data_generator.get_iterator()
    #val_iterator = val_data_generator.get_iterator()


    # build model
    if params.model == 'aan':
        model = AAN_Model(
            params,runconfig,
            train_data_generator.vocab_size,
            train_data_generator.iterator,True)
    elif params.model == 'cbow':
        model = CBOW_Model(
            params,runconfig,
            train_data_generator.vocab_size,
            train_data_generator.iterator,True)
    elif params.model == 'bilstm_sum':
        model = BiLSTM_Sum_Model(
            params,runconfig,
            train_data_generator.vocab_size,
            train_data_generator.iterator,True)
    elif params.model == 'bilstm_concat':
        model = BiLSTM_Concat_Model(
            params,runconfig,
            train_data_generator.vocab_size,
            train_data_generator.iterator,True)
    else:
        raise BaseException('Please select a model.')


    with tf.Session(config=config.session_config) as session:
        session.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(params.logs_dir)
        model_losses = []#[(loss : file_name)]
        for epoch in range(params.epochs):
            # train an epoch
            model.iterator = train_data_generator.iterator
            model.run_epoch(session, model.train_op)

            ### Validation and Save
            #print('Start to validation...')
            #val_loss_f = open(params.save_val_loss,'w')
            #model.iterator = val_iterator
            #val_average_loss = model.run_epoch(session)
            #print('Validation average loss : [%f]'%(val_average_loss))
            #val_loss_f.write('[model]'+params.model+'Train step ['+str(train_step)+'] , val average loss['+str(val_average_loss)+']\n')

            ## Save n best model
            #checkpoint_name = params.model+'-epoch'+str(epoch)+'-model.ckpt'
            #checkpoint_path=os.path.join(params.checkpoints_dir,checkpoint_name)
            #step_ckpt = checkpoint_name+'-'+str(train_step)
            #if len(model_losses) < params.n_best_model:
            #    model_losses.append((val_average_loss,step_ckpt))
            #    model.saver.save(session, checkpoint_path)
            #else:
            #    if model_losses[-1][0] > val_average_loss:#update
            #        # remove
            #        _,del_ckpt=model_losses.pop()
            #        for f in os.listdir(params.checkpoints_dir):
            #            if del_ckpt in f:
            #                r_f=os.path.join(os.path.abspath(params.checkpoints_dir),f)
            #                os.remove(r_f)
            #        model_losses.append((val_average_loss,step_ckpt))
            #        model_losses=sorted(model_losses,key=lambda v:v[0])#sort
            #        model.saver.save(session, checkpoint_path, global_step=int(train_step))
            checkpoint_name = params.model+'-epoch'+str(epoch)+'-model.ckpt'
            checkpoint_path=os.path.join(params.checkpoints_dir,checkpoint_name)
            model.saver.save(session,checkpoint_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.register("type", "bool", lambda v: v.lower() == "true")

    # general
    argparser.add_argument('--batch_size',dest='batch_size',default=128)
    argparser.add_argument('--seq_len',type=int,default=30)
    argparser.add_argument('--val_batch',dest='val_batch',default=128)
    argparser.add_argument('--embedding_size',dest='embedding_size',default=512)
    argparser.add_argument('--clip_grad',type=float,default=5.)
    argparser.add_argument('--embedding_dropout',type=float,default=0.5)
    argparser.add_argument('--learning_rate',type=float,default=0.001)
    argparser.add_argument('--init_type',default='uniform')
    argparser.add_argument('--init_weight',type=float,default=0.2)
    argparser.add_argument('--seed',type=float,default=None)
    argparser.add_argument('--top_k',dest='top_k',default=100)
    # lstm
    argparser.add_argument('--cover_window',type=int,default=1)
    argparser.add_argument('--lstm_dropout',type=float,default=0.5)
    argparser.add_argument('--lstm_hidden_layers',type=int,default=1)
    argparser.add_argument('--lstm_hidden_size',type=int,default=512)
    # cbow
    argparser.add_argument('--window_size',type=int,default=3)
    argparser.add_argument('--num_sampled',dest='num_sampled',default=256)
    # generator
    argparser.add_argument('--shuffle',type='bool',default=True)
    argparser.add_argument('--vocab_threshold',dest='vocab_threshold',default=2)
    argparser.add_argument('--cache_dir',dest='cache_dir',default='cache/')
    argparser.add_argument('--vocab_file',default='data/vocab.txt')
    # train
    argparser.add_argument('--epochs',dest='epochs',default=5)
    argparser.add_argument('--ckpt_epoch',dest='ckpt_epoch',default=4)
    argparser.add_argument('--ckpt_num',dest='ckpt_num',default=15000)
    argparser.add_argument('--logs_dir',dest='logs_dir',default='logs/')
    argparser.add_argument('--checkpoints_dir',default='checkpoints/')
    argparser.add_argument('--n_best_model',default=10)
    argparser.add_argument('--train_dir',default='data/train')
    argparser.add_argument('--val_dir',default='data/val')
    argparser.add_argument('--val_result',default='results/val_result.xls')
    argparser.add_argument('--save_val_loss',default='results/val_loss.txt')
    # use for multihead attention
    argparser.add_argument('--num_hidden_layers',type=int,default=6)
    argparser.add_argument('--hidden_size',type=int,default=512)
    argparser.add_argument('--num_heads',type=int,default=8)
    argparser.add_argument('--attention_dropout',type=float,default=0.0)
    argparser.add_argument('--relu_dropout',type=float,default=0.0)
    argparser.add_argument('--self_attention_type',default="dot_product")
    argparser.add_argument('--max_relative_position',type=int,default=0)
    argparser.add_argument('--layer_preprocess_sequence',default="n")
    argparser.add_argument('--layer_postprocess_sequence',default="da")
    argparser.add_argument('--layer_prepostprocess_dropout',
                           type=float,default=0.1)
    argparser.add_argument('--layer_prepostprocess_dropout_broadcast_dims',
                           default='')
    argparser.add_argument('--norm_type',default="layer")
    argparser.add_argument('--norm_epsilon',type=float,default=1e-06)
    argparser.add_argument('--ffn_layer',default='dense_relu_dense')
    argparser.add_argument('--filter_size',type=int,default=2048)
    argparser.add_argument('--conv_first_kernel',type=int,default=3)
    argparser.add_argument('--parameter_attention_key_channels',
                           type=int,default=0)
    argparser.add_argument('--parameter_attention_value_channels',
                           type=int,default=0)
    argparser.add_argument('--max_length',type=int, default=256)
    argparser.add_argument('--shared_embedding_and_softmax_weights',
                           type='bool',default=False)
    # select model
    argparser.add_argument('--model',default='')

    params = argparser.parse_args()
    config = config
    # Train
    train(params, config)
