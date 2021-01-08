import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
size: (batch_num, (batch_size, trigram_dimension)) e.g.(100, (1024, 30k))
query: Query sample
doc_p: Doc Positive sample
doc_n1, doc_n2, doc_n3, doc_n4: Doc Negative sample
"""

class GRURec(nn.Module):
    def __init__(self, args, hidden=300):
        super(GRURec, self).__init__()
        self.his_len = args.max_hist_length
        self.word_len = args.word_len
        self.neg_number = args.neg_num
        self.hidden = hidden
        self.embed = nn.Embedding(args.word_num, hidden)
        self.news_embed = nn.Embedding(args.news_num, hidden)
        self.gru = nn.GRU(hidden, hidden)
    
    def forward(self, x, mode='train'):
        neg_num = self.neg_number
        if mode == 'test':
            neg_num = 0

        news = x[:, :neg_num + 1 + self.his_len]
        title = x[:, neg_num + 1 + self.his_len:]

        doc = title[:, :self.word_len * (neg_num + 1)]
        his = title[:, self.word_len * (neg_num + 1):]

        his = his.view(-1, self.his_len, self.word_len)
        doc = doc.view(-1, (1 + neg_num), self.word_len)

        doc = self.embed(doc)
        his = self.embed(his)

        doc = doc.mean(dim=-2)
        his = his.mean(dim=-2)

        his = his.permute(1, 0, 2)
        output, hn = self.gru(his)
        user = hn.permute(1, 0, 2).squeeze(1)

        user = user.repeat(1, neg_num + 1).view(-1, neg_num + 1, 300)
        similarity = torch.sum(doc * user, dim=-1)

        return similarity