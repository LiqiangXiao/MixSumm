import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet

INI = 1e-2

# FIXME eccessing 'private members' of pointer net module is bad

class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())
        
        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop
        
    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = (h, c)
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w, cov=None, cov_w=None):
        """ unnormalized attention score"""
        if cov is not None:
            sum_ = attention + torch.mm(query, w) + torch.mm(cov, cov_w)
        else:
            sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w, cov=None, cov_w=None):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w, cov, cov_w ), dim=-1)
        output = torch.mm(score, attention)
        return output

class PtrExtractorRLStop(PtrExtractorRL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
        self._stop = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)

        self._copy_marker = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._copy_marker, -INI, INI)
        
        self._gen_marker = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._gen_marker, -INI, INI)
        
        self._hop_cov = nn.Parameter(torch.Tensor(1, self._lstm_cell.hidden_size))
        init.uniform_(self._hop_cov, -INI, INI)
        
        self._attn_cov = nn.Parameter(torch.Tensor(1, self._lstm_cell.hidden_size))
        init.uniform_(self._attn_cov, -INI, INI)
        
    def forward(self, attn_mem, n_ext=None, step=0, is_coverage=True):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        max_step = attn_mem.size(0)
        attn_mem_copy = attn_mem + self._copy_marker.unsqueeze(0).expand_as(attn_mem)
        attn_mem_gen = attn_mem + self._gen_marker.unsqueeze(0).expand_as(attn_mem)
        attn_mem = torch.cat([attn_mem_copy, attn_mem_gen, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        
        outputs = []
        dists = []
        coverage = torch.zeros(1, attn_mem.size(0)).cuda()
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq,
                                                coverage.t(), self._hop_cov 
                                                )
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq,
                coverage.t(), self._attn_cov 
                ) 
            if not outputs:
                score[0, 2*max_step] = -1e18
            for o in outputs:
                if o.item() < max_step:
                    score[0, o.item()] = -1e18
                    score[0, o.item()+max_step] = -1e18
                else:
                    score[0, o.item()] = -1e18
                    score[0, o.item()-max_step] = -1e18
            if self.training:
                prob = F.softmax(score, dim=-1)
                coverage = coverage + prob
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
                prob = F.softmax(score, dim=-1)
                coverage = coverage + prob
            outputs.append(out)
            if out.item() == max_step*2:                
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (trining)
            return outputs, dists
        else:
            return outputs

class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

#         # regression layer for copy 
#         self._copy_linear = nn.Linear(self._lstm_cell.input_size, 2) 
        
    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        copys = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
#             copy = self._copy_linear(output)
#             copys.append(copy)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(F.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output

class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher

    def forward(self, raw_article_sents, n_abs=None, step=0):
        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._art_enc(enc_sent).squeeze(0)
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_art, step=step)
        else:
            outputs = self._ext(enc_art, n_abs, step=step)
        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_art, n_abs)
            return outputs, scores
        else:
            return outputs
