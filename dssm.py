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

class ForwardNet(nn.Module):
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.l1 = nn.Linear(300, 300)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = nn.Linear(300, 128)
        nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        
        return x

class DSSM(nn.Module):
    def __init__(self, args, hidden=300):
        super(DSSM, self).__init__()
        self.queryNet = ForwardNet()
        self.docNet = ForwardNet()
        self.queryidNet = ForwardNet()
        self.docidNet = ForwardNet()
        self.his_len = args.max_hist_length
        self.word_len = args.word_len
        self.neg_num = args.neg_num
        self.hidden = hidden
        self.embed = nn.Embedding(args.word_num, hidden)
        self.cos = nn.CosineSimilarity(dim=-1)
        self.news_embed = nn.Embedding(args.news_num, hidden)
    
    def forward(self, x, mode='train'):
        neg_num = self.neg_num
        if mode == 'test':
            neg_num = 0

        news = x[:, :neg_num + 1 + self.his_len]
        title = x[:, neg_num + 1 + self.his_len:]

        doc = title[:, :self.word_len * (neg_num + 1)]
        query = title[:, self.word_len * (neg_num + 1):]
        doc_id = news[:, :neg_num + 1]
        query_id = news[:, neg_num + 1:]

        doc = self.embed(doc)
        query = self.embed(query)
        doc_id = self.news_embed(doc_id)
        query_id = self.news_embed(query_id)

        doc = doc.view(-1, neg_num + 1, self.word_len, self.hidden)
        doc = doc.mean(dim=-2)
        doc = doc.view(-1, self.hidden) 
        query = query.mean(dim=-2)

        doc_id = doc_id.view(-1, self.hidden)
        query_id = query_id.mean(dim=-2)

        doc = self.docNet(doc)
        doc = doc.view(-1, neg_num + 1, 128)
        query = self.queryNet(query)
        query = query.repeat(1, neg_num + 1).view(-1, neg_num + 1, 128)

        doc_id = self.docidNet(doc_id)
        doc_id = doc_id.view(-1, neg_num + 1, 128)
        query_id = self.queryidNet(query_id)
        query_id = query_id.repeat(1, neg_num + 1).view(-1, neg_num + 1, 128)
        
        similarity = self.cos(query, doc)
        similarity_id = self.cos(query_id, doc_id)

        return similarity