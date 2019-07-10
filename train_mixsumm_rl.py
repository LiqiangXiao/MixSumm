# load the model 
# build the pipeline 
# main paras and train 
# this version use the plain extractor and rl abstractor
""" abstractor rl training """
import argparse
import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle

from toolz.sandbox.core import unzip
from cytoolz import identity

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic
from model.extract import PtrExtractSumm
from model.copy_summ import CopySumm

from training import BasicTrainer, BasicPipeline
from rl import get_grad_fn
from rl import A2CPipeline
from decoding import load_best_ckpt
from itertools import starmap

from decoding import Abstractor, ArticleBatcher, Extractor, RLExtractor
from metric import compute_rouge_n, compute_rouge_l_summ


# for selfcritic
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch import autograd
from data.batcher import conver2id, pad_batch_tensorize
from utils import PAD, UNK, START, END
from model.copy_summ import CopySumm
from model.extract import ExtractSumm, PtrExtractSumm
from model.rl import ActorCritic
from data.batcher import conver2id, pad_batch_tensorize
from data.data import CnnDmDataset
import math
from time import time
from datetime import timedelta
import numpy as np
from model.util import sequence_mean, len_mask
from cytoolz import concat

# a2c pipeline and actorcritc function 

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')
MAX_ABS_LEN = 30
MAX_EXT = 5
list_cat = lambda x: list(concat(x))

# abstractor model  the agent
class SelfCritic(nn.Module):
    '''rl abstractor'''
    def __init__(self, abstractor, max_len, word2id, cuda=True):
        super().__init__()
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len
        # a lot of copy 
        
    def _prepro(self, raw_article_sents):
        ext_word2id = dict(self._word2id)
        ext_id2word = dict(self._id2word)
        for raw_words in raw_article_sents:
            for w in raw_words:
                if not w in ext_word2id:
                    ext_word2id[w] = len(ext_word2id)
                    ext_id2word[len(ext_id2word)] = w
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        art_lens = [len(art) for art in articles]
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        extend_arts = conver2id(UNK, ext_word2id, raw_article_sents)
        extend_art = pad_batch_tensorize(extend_arts, PAD, cuda=False
                                        ).to(self._device)
        extend_vsize = len(ext_word2id)
        dec_args = (article, art_lens, extend_art, extend_vsize,
                    START, END, UNK, self._max_len)
        return dec_args, ext_id2word

#     self, article, extend_art, extend_vsize, go, eos, unk, max_len
    def forward(self, raw_article_sents):  # for RL abs
        dec_args, id2word = self._prepro(raw_article_sents)
        article, art_lens, extend_art, extend_vsize, go, eos, unk, max_len = dec_args
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        
        #-------gready decode------
        with torch.no_grad():
            self._net.eval()
            decs, attns = self._net.batch_decode(*dec_args)   # gready decode       
            dec_sents_greedy = []
            for i, raw_words in enumerate(raw_article_sents):
                dec = []
                for id_, attn in zip(decs, attns):
                    if id_[i] == END:
                        break
                    elif id_[i] == UNK:
                        dec.append(argmax(raw_words, attn[i]))
                    else:
                        dec.append(id2word[id_[i].item()])
                dec_sents_greedy.append(dec)
        
        #--------batch sampling------
        if self.training:
            self._net.train()
            batch_size = len(art_lens)
            vsize = self._net._embedding.num_embeddings
            attention, init_dec_states = self._net.encode(article, art_lens)
            mask = len_mask(art_lens, attention.device).unsqueeze(-2)
            attention = (attention, mask, extend_art, extend_vsize)
            lstm_in = torch.LongTensor([go]*batch_size).to(article.device)
            outputs = []
            attns = []
            dists = []
            states = init_dec_states
            for i in range(max_len):
                tok, states, attn_score, logit = self._net._decoder.decode_step(
                    lstm_in, states, attention, return_logit=True)
                prob = F.softmax(logit, dim=-1)
                out = torch.multinomial(prob, 1).detach()
                if i == 0:
                    flag = (out!=eos)
                else:
                    flag = flag*(out!=eos)
                dist = torch.log( prob.gather(1, out) )
                dists.append(dist)
                attns.append(attn_score)
                outputs.append(out[:, 0].clone())
                if flag.sum().item() ==0:
                    break 
                lstm_in = out.masked_fill(out >= vsize, unk)

            output_word_batch =[] 
            dist_batch = []
            for i, raw_words in enumerate(raw_article_sents):
                words = []
                diss  = []
                for id_, attn, dis in zip(outputs, attns, dists):
                    diss.append(dis[i:i+1])
                    if id_[i] == END:
                        break
                    elif id_[i] == UNK:
                        words.append(argmax(raw_words, attn[i]))
                    else:
                        words.append(id2word[id_[i].item()])
                output_word_batch.append(words)
                dist_batch.append( sum(diss) )
        
        if self.training:
            return dec_sents_greedy, output_word_batch, dist_batch
        else:
            return dec_sents_greedy

def sc_train_step(agent, extractor, loader, opt, grad_fn, reward_fn=compute_rouge_l_summ, gamma=0.95):
    opt.zero_grad()
    list_cat = lambda x: list(concat(x))
    art_batch, abs_batch, ext_batch = next(loader)
    # extract the art_batch 
    with torch.no_grad():
        ext_sents = []
        sum_sents = []
        indices = []
        for raw_arts in art_batch:
            inds = extractor(raw_arts)
            inds = [i.item() for i in inds]
            indices.append(inds)
            ext_sents += [raw_arts[idx]
                         for idx in inds if idx < len(raw_arts)]
            sum_sents += [raw_arts[idx-len(raw_arts)]
                          for idx in inds if len(raw_arts)<=idx<len(raw_arts)*2]
    #--------batch decode--------
    if sum_sents:
        dec_greedy, outputs, dists = agent(sum_sents)
        count_ext= 0
        count_sum= 0
        count_dists = 0 
        dec_greedy_mix = []
        outputs_mix = []
        dists_mix = []
        for indice in indices:
            max_step = indice[-1]/2
            for ind in indice:
                if ind<max_step:
                    dec_greedy_mix.append(ext_sents[count_ext])
                    outputs_mix.append(ext_sents[count_ext])
                    count_ext+=1
                    dists_mix.append(0)
                elif max_step<=ind<max_step*2:
                    dec_greedy_mix.append(dec_greedy[count_sum])
                    outputs_mix.append(outputs[count_sum])
                    count_sum+=1
                    dists_mix.append(dists[count_dists])
                    count_dists+=1
    else:
        dec_greedy_mix, outputs_mix, dists = ext_sents, [0]*len(ext_sents), [0]*len(ext_sents)
    assert  len(dec_greedy_mix) == len(outputs_mix) == len(dists_mix)

    ave_rewards = 0
    ave_baselines = 0 
    advantage_batch = []

    i = 0
#     # replacement reward 
#     for ex, abss in zip( indices, abs_batch):
#         ex = ex[:-1]
#         baseline =  reward_fn(dec_greedy_mix[i:i+len(ex)], abss)
#         ave_baselines += baseline
#         rewards = [ reward_fn( dec_greedy_mix[i:i+num] + dec_greedy_mix[i+num+1:i+len(ex)] + outputs_mix[i+num:i+num+1], abss) 
#                    for num in range(len(ex)) ]
#         ave_rewards += sum(rewards)/len(rewards)
#         advantage = [ r - baseline for r in rewards]
#         advantage_batch += advantage 
#         i += len(ex)
#      plain reward 
    for ex, abss in zip( indices, abs_batch):
        ex = ex[:-1]
        baseline =  reward_fn(dec_greedy_mix[i:i+len(ex)], abss)
        ave_baselines += baseline
        rewards = [reward_fn(outputs_mix[i:i+len(ex)], abss)]*len(ex)
        ave_rewards += sum(rewards)/len(rewards)
        advantage = [ r - baseline for r in rewards]
        advantage_batch += advantage 
        i += len(ex)
    assert i == len(dec_greedy_mix) 
    
    advantage = torch.Tensor(advantage_batch).cuda()
    ave_advantage = advantage.mean().item()
    losses = [ -d*(a) for a, d in zip(advantage, dists_mix ) if d is not 0] 
    losses = [loss/len(losses) for loss in losses]
    # backprop and update
    autograd.backward(
         losses,
        [torch.ones(1).cuda()]*(len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = ave_rewards/len(art_batch)
    log_dict['baseline'] = ave_baselines/len(art_batch)
    log_dict['advantage'] = ave_advantage
#     log_dict['advantage'] = avg_advantage.item()/len(indices)
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict

def sc_validate(agent, extractor, loader):
    agent.eval()
    start = time()
    print('start running validation...')
    ave_reward = 0
    ave_reward_1 = 0 
    ave_reward_2 = 0 
    batch_num = 0 
    for art_batch, abs_batch, ext_batch in loader:
        ext_sents = []
        sum_sents = []
        indices = []
        for raw_arts in art_batch:
            inds = extractor(raw_arts)
            inds = [i.item() for i in inds]
            indices.append(inds)
            ext_sents += [raw_arts[idx]
                         for idx in inds if idx < len(raw_arts)]
            sum_sents += [raw_arts[idx-len(raw_arts)]
                          for idx in inds if len(raw_arts)<=idx<len(raw_arts)*2]
        #--------batch decode--------
        if sum_sents:
            dec_greedy = agent(sum_sents)
            count_ext= 0
            count_sum= 0
            dec_greedy_mix = []
            for indice in indices:
                max_step = indice[-1]/2
                for ind in indice:
                    if ind<max_step:
                        dec_greedy_mix.append(ext_sents[count_ext])
                        count_ext+=1
                    elif max_step<=ind<max_step*2:
                        dec_greedy_mix.append(dec_greedy[count_sum])
                        count_sum+=1
        else:
            dec_greedy_mix = ext_sents
        i = 0
        for ex, abss in zip(indices, abs_batch):
            ex=ex[:-1]
            ave_reward += compute_rouge_l_summ(dec_greedy_mix[i:i+len(ex)], abss)
            ave_reward_1 += compute_rouge_n( list_cat(dec_greedy_mix[i:i+len(ex)]), list(concat(abss)), n=1)
            ave_reward_2 += compute_rouge_n( list_cat(dec_greedy_mix[i:i+len(ex)]), list(concat(abss)), n=2)
            i += len(ex)    
            batch_num += 1
        assert i == len(dec_greedy_mix)
    ave_reward /= (batch_num/100)
    ave_reward_1 /=(batch_num/100)
    ave_reward_2 /=(batch_num/100)
    print('finished in {}! avg reward: {:.2f} rouge-1: {:.2f} rouge-2: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), ave_reward, ave_reward_1, ave_reward_2))
    return {'reward': ave_reward}

class SCPipeline(BasicPipeline):
    def __init__(self, name,
                 net, extractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn
        self._extractor = extractor
        self._reward_fn = reward_fn
        self._gamma = gamma
    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = sc_train_step(
            self._net, self._extractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._reward_fn,
            self._gamma
        )
        return log_dict

    def validate(self):
        return sc_validate(self._net, self._extractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

    
# ----------traning part-----------
    
class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        art_exts = js_data['extracted']
        return art_sents, abs_sents, art_exts

def load_abs_net(abs_dir):
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    abs_args = abs_meta['net_args']
    abs_ckpt = load_best_ckpt(abs_dir)
    word2id = pkl.load(open(join(abs_dir, 'vocab.pkl'), 'rb'))
    abstractor = CopySumm(**abs_args)
    abstractor.load_state_dict(abs_ckpt)
    return abstractor, word2id

def configure_net(abs_dir, ext_dir, cuda):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    extractor  = RLExtractor(ext_dir,  cuda)
    # load ML trained extractor net and buiild RL agent
    abstractor, agent_vocab = load_abs_net(abs_dir)
#     agent = ActorCritic(extractor._sent_enc,
#                         extractor._art_enc,
#                         extractor._extractor,
#                         ArticleBatcher(agent_vocab, cuda))
    agent = SelfCritic(abstractor,
                       MAX_ABS_LEN, agent_vocab, cuda=cuda)
    if cuda:
        agent = agent.cuda()
    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, extractor, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    train_params['gamma']          = gamma
    train_params['reward']         = reward
    train_params['stop_coeff']     = stop_coeff
    train_params['stop_reward']    = stop_reward
    return train_params


def build_batchers(batch_size):
    def coll(batch):
        art_batch, abs_batch, ext_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents, ext_batch
    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader


def train(args):
    if not exists(args.path):
        os.makedirs(args.path)
    # make net
    agent, agent_vocab, extractor, net_args = configure_net(
        args.abs_dir, args.ext_dir, args.cuda)

    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    train_batcher, val_batcher = build_batchers(args.batch)
    
    # TODO different reward
    def multi_reward(x,y):
        return ( compute_rouge_l_summ(x,y)
                + compute_rouge_n(list_cat(x),list_cat(y),n=2) 
                + 1/3*compute_rouge_n(list_cat(x),list_cat(y),n=1) )
#     reward_fn = compute_rouge_l_summ
    reward_fn=multi_reward
    
    # save extractor binary
    if args.ext_dir is not None:
        ext_ckpt = {}
        ext_ckpt['state_dict'] = load_best_ckpt(args.ext_dir, reverse=True)
        ext_vocab = pkl.load(open(join(args.ext_dir, 'agent_vocab.pkl'), 'rb'))
        
        ext_dir = join(args.path, 'extractor')
        os.makedirs(join(ext_dir, 'ckpt'))
        with open(join(ext_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['extractor'], f, indent=4)
        torch.save(ext_ckpt, join(ext_dir, 'ckpt/ckpt-0-0'))
        with open(join(ext_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(ext_vocab, f)
            
    # save configuration
    meta = {}
    meta['net']           = 'rnn-ext_abs_rl'
    meta['net_args']      = net_args
    meta['train_params']  = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=0.5e-4,
                                  patience=args.lr_p)
    
    pipeline = SCPipeline(meta['net'], agent, extractor,
                           train_batcher, val_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score')
    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--dev', action='store', default = '0',
                        help='gpu number')


    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')
    parser.add_argument('--plan', type=int, action='store', default=None,
                        help='reward plan')
    
    # training options
    parser.add_argument('--reward', action='store', default='rouge-l',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=3,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    train(args)