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
    def __init__(self, args):
        super(DSSM, self).__init__()
        self.queryNet = ForwardNet()
        self.docNet = ForwardNet()
        self.his_len = args.max_hist_length
        self.word_len = args.word_len
        self.neg_num = args.neg_num
        self.embed = nn.Embedding(args.word_num, 300)
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, mode='train'):
        neg_num = self.neg_num
        if mode == 'test':
            neg_num = 0

        doc = x[:, :self.word_len * (neg_num + 1)]
        query = x[:, self.word_len * (neg_num + 1):]

        doc = self.embed(doc)
        query = self.embed(query)

        doc = doc.view(-1, neg_num + 1, self.word_len, 300)
        doc = doc.mean(dim=-2)
        doc = doc.view(-1, 300) 
        query = query.mean(dim=-2)

        doc = self.docNet(doc)
        doc = doc.view(-1, neg_num + 1, 128)
        query = self.queryNet(query)
        query = query.repeat(1, neg_num + 1).view(-1, neg_num + 1, 128)
        
        similarity = self.cos(query, doc)

        return similarity