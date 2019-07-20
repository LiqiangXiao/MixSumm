""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
import re
import pickle as pkl
from itertools import starmap
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize
from data.batcher import conver2id, pad_batch_tensorize
from data.data import CnnDmDataset

from model.copy_summ import CopySumm
from model.extract import PtrExtractSumm
from decoding import DecodeDataset, load_best_ckpt, ArticleBatcher
from decoding import make_html_safe, _process_beam
from utils import PAD, UNK, START, END
from train_abs_rl import SelfCritic
from model.rl import ActorCritic

from model.beam_search import _Hypothesis
import numpy as np


class RLExtractor(object):
    def __init__(self, ext_dir, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        assert ext_meta['net'] == 'rnn-ext_abs_rl'
        ext_args = ext_meta['net_args']['extractor']['net_args']
        word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        extractor = PtrExtractSumm(**ext_args)
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(word2id, cuda))
        ext_ckpt = load_best_ckpt(ext_dir, reverse=True)
        agent.load_state_dict(ext_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = agent.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}

    def __call__(self, raw_article_sents):
        self._net.eval()
        indices = self._net(raw_article_sents)
        return indices

class RLAbstractor(object):
    def __init__(self, abs_dir, max_len=30, cuda=True):
        abs_meta = json.load(open(join(abs_dir, 'meta.json')))
        assert abs_meta['net'] == 'rnn-ext_abs_rl'
        abs_args = abs_meta['net_args']['abstractor']['net_args']
        abs_ckpt = load_best_ckpt(abs_dir, reverse=True)
        word2id = pkl.load(open(join(abs_dir, 'agent_vocab.pkl'), 'rb'))
        abstor = CopySumm(**abs_args)
        abstractor = SelfCritic(abstor, max_len, word2id, cuda)
        abstractor.load_state_dict(abs_ckpt)
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = abstractor.to(self._device)
        self._word2id = word2id
        self._id2word = {i: w for w, i in word2id.items()}
        self._max_len = max_len

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

    def __call__(self, raw_article_sents):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        decs, attns = self._net._net.batch_decode(*dec_args)
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for id_, attn in zip(decs, attns):
                if id_[i] == END:
                    break
                elif id_[i] == UNK:
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents

class RLBeamAbstractor(RLAbstractor):
    def __call__(self, raw_article_sents, beam_size=5, diverse=1.0):
        self._net.eval()
        dec_args, id2word = self._prepro(raw_article_sents)
        dec_args = (*dec_args, beam_size, diverse)
        all_beams = self._net._net.batched_beamsearch(*dec_args)
        all_beams = list(starmap(_process_beam(id2word),
                                 zip(all_beams, raw_article_sents)))
        return all_beams


def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = RLAbstractor(model_dir,
                                    max_len, cuda)
        else:
            abstractor = RLBeamAbstractor(model_dir,
                                        max_len, cuda)
    extractor = RLExtractor(join(model_dir, 'extractor'), cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_sents = []
            sum_sents =[]
            ext_inds = []
            all_indices = []
            ext_inds_rerank =[]
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)  # exclude EOE
                ext = [i.item() for i in ext]
                ext_inds += [(len(ext_sents)+len(sum_sents), len(ext)-1)]
                all_indices.append(ext)
                ext_sents += [raw_art_sents[idx]
                             for idx in ext if idx < len(raw_art_sents)]
                sum_len = len(sum_sents)
                sum_sents += [raw_art_sents[idx-len(raw_art_sents)]
                              for idx in ext if len(raw_art_sents)<=idx<len(raw_art_sents)*2]
                ext_inds_rerank += [(sum_len, len(sum_sents)-sum_len) ] 
            if sum_sents:
                if beam_size > 1:
                    all_beams = abstractor(sum_sents, beam_size, diverse)
                    # mix exts and summ with beam_size 
                    count_ext= 0
                    count_sum= 0
                    mix_beams = []
                    for indice in all_indices:
                        max_step = indice[-1]/2
                        for ind in indice:
                            if ind<max_step:
                                mix_beams.append( [ _Hypothesis(ext_sents[count_ext], 
                                                               0,0,0) ]*beam_size)
                                count_ext+=1
                            elif max_step<=ind<max_step*2:
                                mix_beams.append(all_beams[count_sum])
                                count_sum+=1
                    dec_outs = rerank_mp(mix_beams, ext_inds)
                    assert len(dec_outs) == len(mix_beams)  
                else:
                    sum_sents = abstractor(sum_sents)
                    # mix exts and summs
                    count_ext= 0
                    count_sum= 0
                    dec_outs = []
                    for indice in all_indices:
                        max_step = indice[-1]/2
                        for ind in indice:
                            if ind<max_step:
                                dec_outs.append(ext_sents[count_ext])
                                count_ext+=1
                            elif max_step<=ind<max_step*2:
                                dec_outs.append(sum_sents[count_sum])
                                count_sum+=1
                    
            else:
                dec_outs = ext_sents
            assert len(dec_outs) == len(ext_sents) + len(sum_sents) 
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
    print()

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams, top_lens=5):
    # length rank 
    beams_tmp = []
    for beam in beams:
        beam_len = list(map(lambda x: len(x.sequence), beam))
        beam = [beam[i] for i in np.argsort(beam_len)[-top_lens:] ]
        beams_tmp.append(beam)
    beams = beams_tmp 
    
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')
    parser.add_argument('--dev', action='store', default = '0',
                        help='gpu number')
    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda)
