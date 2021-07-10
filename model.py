
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
from torch import nn

class BertSpanNER(nn.Module):
    def __init__(self, encoder, hid_dim):
        super().__init__()
        # self.encoder = encoder.to('cpu')
        # if model == 'distilbertbase':
        #     self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # else:
        #     raise "Not Implemented"
        self.encoder = encoder
        self.ner_classifier = nn.Sequential(
            nn.Linear(hid_dim * 2 + 0, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 8)
        )
        
    def forward(self, seq, mask, span_batch, span_index):
        # print(seq.shape)
        embs = self.encoder(seq, attention_mask=mask)[0][:, 1:, :]
        emb_ls = []
        for i, (j, k) in enumerate(span_index):
            # print(i, j, k)
            # j, k : the start index and length for spans of this batch
            spans = span_batch[j: j+k]  # (n_span, 2)
            x = embs[i, spans[:, 0], :]  # (n_span, 768)
            y = embs[i, spans[:, 1], :]
            # y = embs[i, span_batch[j+k][0]: span_batch[j+k][0] + span_batch[j+k][1], :]
            emb_ls.append(torch.cat((x, y), dim=-1))
        emb_ls = torch.cat(emb_ls, dim=0)
        return self.ner_classifier(emb_ls)
    
class BertRE(nn.Module):
    def __init__(self, encoder, hid_dim):
        super().__init__()
        self.encoder = encoder
        self.ner_classifier = nn.Sequential(
            nn.Linear(hid_dim * 4 + 0, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 8)
        )
        
    def forward(self, seq, mask, pair_batch, pair_index):
        # print(seq.shape)
        embs = self.encoder(seq, attention_mask=mask)[0][:, 1:, :]
        emb_ls = []
        for i, (j, k) in enumerate(pair_index):
            # print(i, j, k)
            # j, k : the start index and length for spans of this batch
            pair = pair_batch[j: j+k]  # (n_pair, 6)
            h_s = embs[i, pair[:, 0], :]  # (n_span, 768)
            h_t = embs[i, pair[:, 1], :]
            t_s = embs[i, pair[:, 2], :]
            t_t = embs[i, pair[:, 3], :]
            # y = embs[i, span_batch[j+k][0]: span_batch[j+k][0] + span_batch[j+k][1], :]
            emb_ls.append(torch.cat((h_s, h_t, t_s, t_t), dim=-1))
        emb_ls = torch.cat(emb_ls, dim=0)
        return self.ner_classifier(emb_ls)