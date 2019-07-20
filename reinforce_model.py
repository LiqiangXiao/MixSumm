""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ
from training import BasicPipeline

def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    batch_num = 0 
    copy_rate = 0
    with torch.no_grad():
        for art_batch, abs_batch in loader:
            ext_sents = []
            sum_sents = []
            ext_inds = []
            all_indices = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                ext_inds += [(len(ext_sents)+len(sum_sents), len(indices)-1)]
                all_indices.append(indices)
                ext_sents += [raw_arts[idx.item()]
                             for idx in indices if idx.item() < len(raw_arts)]
                sum_sents += [raw_arts[idx.item()-len(raw_arts)]
                              for idx in indices if len(raw_arts)<=idx.item()<len(raw_arts)*2]
            if sum_sents:
                sum_sents = abstractor(sum_sents)
                # mix ext and sum
                count_ext= 0
                count_sum= 0
                all_summs = []
                for indice in all_indices:
                    max_step = indice[-1].item()/2
                    for ind in indice:
                        if ind.item()<max_step:
                            all_summs.append(ext_sents[count_ext])
                            count_ext+=1
                        elif max_step<=ind.item()<max_step*2:
                            all_summs.append(sum_sents[count_sum])
                            count_sum+=1
            else:
                all_summs = ext_sents
            assert len(all_summs) == len(ext_sents)+len(sum_sents)
            copy_rate += len(ext_sents)/float(len(all_summs))
            batch_num += 1
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n] 
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    copy_rate /= (batch_num/100)
    print('finished in {}! avg reward: {:.3f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    print('copy rate:{:.6f}'.format(copy_rate) )
    
    return {'reward': avg_reward, 'val_copy_rate': copy_rate}

def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0, step=0, reward_plan=1):
    opt.zero_grad()
    indices = []
    probs = []
    baselines = []
    ext_sents = []
    sum_sents = []
    art_batch, abs_batch = next(loader)
    for raw_arts in art_batch:
        (inds, ms), bs = agent(raw_arts, step=step)
        baselines.append(bs)
        indices.append(inds)
        probs.append(ms)
        ext_sents += [raw_arts[idx.item()]
                     for idx in inds if idx.item() < len(raw_arts)]
        sum_sents += [raw_arts[idx.item()-len(raw_arts)]
                       for idx in inds if len(raw_arts)<=idx.item()<len(raw_arts)*2]
    if sum_sents:
        with torch.no_grad():
             sum_sents= abstractor(sum_sents)
        # mix the exts and sums
        count_ext= 0
        count_sum= 0
        summaries = []
        for indice in indices:
            max_step = indice[-1].item()/2
            for ind in indice[:-1]:
                if ind.item()<max_step:
                    summaries.append(ext_sents[count_ext])
                    count_ext +=1
                elif max_step<=ind.item()<max_step*2:
                    summaries.append(sum_sents[count_sum])
                    count_sum+=1
    else:
        summaries = ext_sents
    assert len(summaries) == len(ext_sents)+len(sum_sents)
    copy_rate = len(ext_sents)/len(summaries)
    
    i = 0
    rewards = []
    avg_reward = 0
    for inds, abss in zip(indices, abs_batch):
        # plain 
        if reward_plan == 0: 
            reward_save = [ compute_rouge_l_summ(summaries[i:i+j+1], abss) for j in range(min(len(inds)-1, len(abss))) ]
            rs = ([ (reward_save[j] - reward_save[j-1]) if j>0 else reward_save[j]
                    for j in range(min(len(inds)-1, len(abss)))]
                  + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
                  + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))])
        # margin 
        elif reward_plan ==1:
            reward_save = [ compute_rouge_l_summ(summaries[i:i+j+1], abss) for j in range(len(inds)-1) ]
            rs = ([ (reward_save[j] - reward_save[j-1]) if j>0 else reward_save[j]
                    for j in range(len(inds)-1) ]
                  + [stop_coeff*stop_reward_fn(
                      list(concat(summaries[i:i+len(inds)-1])),
                      list(concat(abss)))])
        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
    indices = list(concat(indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    
    # standardize rewards
    reward = torch.Tensor(rewards).to(baselines[0].device)
    reward = (reward - reward.mean()) / (
        reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for action, p, r, b in zip(indices, probs, reward, baseline):
        advantage = r - b
        avg_advantage += advantage
        losses.append(-p.log_prob(action)
                      * (advantage/len(indices)) )   # divide by T*B  + 1e-2*torch.exp(p.log_prob(action))*p.log_prob(action)
    critic_loss = F.mse_loss(baseline, reward)
     # backprop and update
    autograd.backward(
        [critic_loss] + losses,
        [torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    log_dict['copy_rate'] = copy_rate
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict

def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self, step=0):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff, 
            step=step
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing
