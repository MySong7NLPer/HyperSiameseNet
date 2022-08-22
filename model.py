import torch
from torch import nn
from transformers import BertModel, RobertaModel
import geoopt as gt
import math

class MatchSum(nn.Module):

    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('../MatchSum/transformers_model/bert-base-uncased',
                                                     output_hidden_states=True)
        else:
            self.encoder = RobertaModel.from_pretrained('../MatchSum/transformers_model/roberta-base',
                                                        output_hidden_states=True)
        self.ball = gt.PoincareBall()
        self.rank = 128
        self.trans_d = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.trans_s = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.trans_c = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}
    def forward(self, text_id, candidate_id, summary_id):

        batch_size = text_id.size(0)
        pad_id = 0  # for BERT
        if text_id[0][0] == 0:
            pad_id = 1  # for RoBERTa
        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0]
        doc_emb = self.ball.expmap0(self.trans_d(out[:, 0, :]))

        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0]  # last layer
        summary_emb = self.ball.expmap0(self.trans_s(out[:, 0, :]))
        summary_score = -self.ball.dist2(summary_emb, doc_emb)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = self.ball.expmap0(self.trans_c(out[:, 0, :])).view(batch_size, candidate_num,self.rank)  # [batch_size, candidate_num, hidden_size]
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = -self.ball.dist2(candidate_emb, doc_emb) # [batch_size, candidate_num]
        return {'score': score, 'summary_score': summary_score}

    def mean_max_pooling(self, relevance):
        # emb = [batch, doc_length, doc_length]
        max_signals, _ = torch.max(relevance, -1)
        mean_signals = torch.mean(relevance, -1).unsqueeze(-1)
        # mean_signals = [batch, doc_length, 1]
        return torch.cat([max_signals.unsqueeze(-1), mean_signals], -1)  # [batch, doc_length, 2]

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def klein_constraint(self, x):
        last_dim_val = x.size(-1)
        norm = torch.reshape(torch.norm(x, dim=-1), [-1, 1])
        maxnorm = (1 - self.eps[x.dtype])
        cond = norm > maxnorm
        x_reshape = torch.reshape(x, [-1, last_dim_val])
        projected = x_reshape / (norm + self.min_norm) * maxnorm
        x_reshape = torch.where(cond, projected, x_reshape)
        x = torch.reshape(x_reshape, list(x.size()))
        return x

    def to_klein(self, x, c=1):
        x_2 = torch.sum(x * x, dim=-1, keepdim=True)
        x_klein = 2 * x / (1.0 + x_2)
        x_klein = self.klein_constraint(x_klein)
        return x_klein

    def klein_to_poincare(self, x, c=1):
        x_poincare = x / (1.0 + torch.sqrt(1.0 - torch.sum(x * x, dim=-1, keepdim=True)))
        x_poincare = self.proj(x_poincare, c)
        return x_poincare

    def lorentz_factors(self, x):
        x_norm = torch.norm(x, dim=-1)
        return 1.0 / (1.0 - x_norm ** 2 + self.min_norm)

    def einstein_midpoint(self, x, c=1):
        x = self.to_klein(x, c)
        x_lorentz = self.lorentz_factors(x)
        x_norm = torch.norm(x, dim=-1)
        # deal with pad value
        x_lorentz = (1.0 - torch._cast_Float(x_norm == 0.0)) * x_lorentz
        x_lorentz_sum = torch.sum(x_lorentz, dim=-1, keepdim=True)
        x_lorentz_expand = torch.unsqueeze(x_lorentz, dim=-1)
        x_midpoint = torch.sum(x_lorentz_expand * x, dim=1) / x_lorentz_sum
        x_midpoint = self.klein_constraint(x_midpoint)
        x_p = self.klein_to_poincare(x_midpoint, c)
        return x_p
