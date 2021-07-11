
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
            spans = span_batch[j: j+k].long()  # (n_span, 2)
            x = embs[i, spans[:, 0], :]  # (n_span, 768)
            y = embs[i, spans[:, 1], :]
            # y = embs[i, span_batch[j+k][0]: span_batch[j+k][0] + span_batch[j+k][1], :]
            emb_ls.append(torch.cat((x, y), dim=-1))
        emb_ls = torch.cat(emb_ls, dim=0)
        return self.ner_classifier(emb_ls)
    
class BertRE(nn.Module):
    def __init__(self, encoder, hid_dim, att):
        super().__init__()
        self.encoder = encoder
        self.agg = nn.Linear(hid_dim * 4 + 0, 768)
        '''
        self.re_classifier = nn.Sequential(
            nn.Linear(hid_dim * 4, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 8)
        
        '''
        self.relu = nn.ReLU()
        self.attention = att

        self.dropout = nn.Dropout(0.2)
        if att:
            self.re_cls = nn.Linear(768 * 2, 8)
            self.multihead = nn.MultiheadAttention(768, 3, dropout=0.2, batch_first=True)
        else:
            self.re_cls = nn.Linear(768, 8)
            
    def forward(self, seq, mask, pair_batch, pair_index):
        # print(seq.shape)
        embs = self.encoder(seq, attention_mask=mask)[0]
        # refined_embs = embs[:, 1:, :]
        emb_ls = []
        context_ls = []
        mask_ls = []
        # If a BoolTensor is provided, the positions with the value of True
        # will be ignored while the position with the value of False will be unchanged
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html?highlight=attention#torch.nn.MultiheadAttention
        for i, (j, k) in enumerate(pair_index):
            # print(i, j, k)
            # j, k : the start index and length for spans of this batch
            pair = pair_batch[j: j+k].long()  # (n_pair, 6)
            h_s = embs[i, 1 + pair[:, 0], :]  # (n_span, 768)
            h_t = embs[i, 1 + pair[:, 1], :]
            t_s = embs[i, 1 + pair[:, 2], :]
            t_t = embs[i, 1 + pair[:, 3], :]
            # y = embs[i, span_batch[j+k][0]: span_batch[j+k][0] + span_batch[j+k][1], :]
            context_ls.append(embs[i].unsqueeze(0).expand(k, -1, -1))  # ls of (n_entity, L_key, emb_size)
            mask_ls.append(1 - mask[i].unsqueeze(0).expand(k, -1))
            emb_ls.append(torch.cat((h_s, h_t, t_s, t_t), dim=-1))  # (n_entity, 4*dim)
        emb_ls = torch.cat(emb_ls, dim=0)  # N_batch = 0
        entity_pair = self.relu(self.agg(emb_ls).view(-1, 1, 768))  # (N_batch, 1, 768)
        mask_ls = torch.cat(mask_ls, dim=0)  # (N_batch, L_key)
        context_ls = torch.cat(context_ls, dim=0) # (N_batch, L_key, emb_size)
        if self.attention:
            output, attn = self.multihead(entity_pair, context_ls, context_ls, key_padding_mask=mask_ls)  # (N_batch, 1, 768)
            return self.re_cls(torch.cat((output[:, 0, :], self.dropout(entity_pair[:, 0, :])), dim=1))
        # return self.ner_classifier(emb_ls)
        else:
            return self.re_cls(self.dropout(entity_pair[:, 0, :]))