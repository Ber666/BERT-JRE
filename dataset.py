import json
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import *

from collections import Counter

ner_list = ["N/A", 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
re_list = ["N/A", 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE']

def re_collate_fn(batch):
    seq = pad_sequence([torch.tensor(i['tokens_id']) for i in batch], padding_value=0, batch_first=True).to('cuda')
    mask = (seq > 0).float()
    pair_index = torch.zeros(len(batch), 2)
    cur_len = 0
    pair_batch = []
    pair_label_batch = []
    for i, j in enumerate(batch):
        pair_batch += j['entity_pairs']
        pair_index[i][0] = cur_len
        tmp_len = len(j['entity_pairs'])
        pair_index[i][1] = tmp_len
        cur_len += tmp_len
        pair_label_batch += j['entity_pairs_label']
        
    pair_batch = torch.tensor(pair_batch).to('cuda').long()
    pair_index = pair_index.to('cuda').long()
    pair_label_batch = torch.tensor(pair_label_batch).to('cuda')
    # spans = torch.tensor(inputs['spans'])
    return seq, mask, pair_batch, pair_label_batch, pair_index

def jre_collate_fn(batch):
    seq = pad_sequence([torch.tensor(i['tokens_id']) for i in batch], padding_value=0, batch_first=True).to('cuda')
    mask = (seq > 0).float()
    span_index = torch.zeros(len(batch), 2)
    cur_len_ner = 0
    pair_index = torch.zeros(len(batch), 2)
    cur_len = 0
    span_batch = []
    span_label_batch = []
    relations = []
    pair_batch = []
    pair_label_batch = []
    for i, j in enumerate(batch):
        pair_batch += j['entity_pairs']
        pair_index[i][0] = cur_len
        tmp_len = len(j['entity_pairs'])
        pair_index[i][1] = tmp_len
        cur_len += tmp_len
        pair_label_batch += j['entity_pairs_label']
        
        span_batch += j['spans']
        span_index[i][0] = cur_len_ner
        tmp_len = len(j['spans'])
        span_index[i][1] = tmp_len
        cur_len_ner += tmp_len
        span_label_batch += j['spans_label']
        relations.append(j['relations'])
        
    pair_batch = torch.tensor(pair_batch).to('cuda').long()
    pair_index = pair_index.to('cuda').long()
    pair_label_batch = torch.tensor(pair_label_batch).to('cuda')
        
    span_batch = torch.tensor(span_batch).to('cuda').long()
    span_index = span_index.to('cuda').long()
    span_label_batch = torch.tensor(span_label_batch).to('cuda')
    # spans = torch.tensor(inputs['spans'])
    return seq, mask, span_batch, span_label_batch, span_index, pair_batch, pair_label_batch, pair_index

def ner_collate_fn(batch):
    seq = pad_sequence([torch.tensor(i['tokens_id']) for i in batch], padding_value=0, batch_first=True).to('cuda')
    mask = (seq > 0).float()
    span_index = torch.zeros(len(batch), 2)
    cur_len = 0
    span_batch = []
    span_label_batch = []
    for i, j in enumerate(batch):
        span_batch += j['spans']
        span_index[i][0] = cur_len
        tmp_len = len(j['spans'])
        span_index[i][1] = tmp_len
        cur_len += tmp_len
        span_label_batch += j['spans_label']
    span_batch = torch.tensor(span_batch).to('cuda').long()
    span_index = span_index.to('cuda').long()
    span_label_batch = torch.tensor(span_label_batch).to('cuda')
    # spans = torch.tensor(inputs['spans'])
    return seq, mask, span_batch, span_label_batch, span_index

class JREDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_span_len):
        # super().__init__()
        with open(data_path, 'r') as f:
            self.raw_data = [sent for sent in json.load(f) if len(sent['tokens']) > 0]
        self.ner_id2label = ner_list
        self.ner_label2id = {j:i for i, j in enumerate(ner_list)}
        self.re_id2label = re_list
        self.re_label2id = {j:i for i, j in enumerate(re_list)}
        self.data = []
        c_ner, c_re, c_span_len = Counter(), Counter(), Counter()
        self.c_ori_ner = 0
        self.c_ori_re = 0
        for l in tqdm(self.raw_data):
            sub_token_mapping = []  # (index, len)
            refined_tokens = []
            cnt = 0
            # first re-tokenize the tokens with BertTokenizer
            self.c_ori_ner += len(l['entities'])
            self.c_ori_re += len(l['relations'])
            for t in l['tokens']:
                subtokens = tokenizer.tokenize(t)
                tmp_len = len(subtokens)
                refined_tokens += subtokens
                sub_token_mapping.append((cnt, tmp_len))
                cnt += tmp_len
            
            refined_entities = {(sub_token_mapping[e[0]][0], sub_token_mapping[e[1]-1][0] + sub_token_mapping[e[1]-1][1]): e[2] for e in l['entities']}
            refined_relations = {(sub_token_mapping[r[0]][0], sub_token_mapping[r[1]-1][0] + sub_token_mapping[r[1]-1][1], \
                sub_token_mapping[r[2]][0], sub_token_mapping[r[3]-1][0] + sub_token_mapping[r[3]-1][1]): r[4] for r in l['relations']}
            c_span_len += Counter([j - i for i, j in refined_entities])
            spans, spans_label = [], []
            # span2id = {}
            cnt = 0
            for i in range(len(refined_tokens)):
                for j in range(i + 1, min(len(refined_tokens), i + max_span_len + 1)):
                    spans.append((i, j))
                    # span2id[(i, j)] = cnt
                    cnt += 1
                    spans_label.append(self.ner_label2id[refined_entities.get((i, j), 'N/A')])
            entity_pairs, entity_pairs_label = [], []
            for s, s_ in refined_entities.items():
                for t, t_ in refined_entities.items():
                    # if j >= i:
                    #     break
                    # print(s, s_, t, t_)
                    entity_pairs.append((s[0], s[1], t[0], t[1], self.ner_label2id[s_], self.ner_label2id[t_]))
                    entity_pairs_label.append(self.re_label2id[refined_relations.get((s[0], s[1], t[0], t[1]), 'N/A')])
            refined_tokens_ids = tokenizer.convert_tokens_to_ids(refined_tokens)
            self.data.append({
                'tokens': refined_tokens,
                'tokens_id': [tokenizer.cls_token_id, *refined_tokens_ids, tokenizer.sep_token_id],
                'entities': refined_entities,
                'relations': refined_relations,
                'spans': spans,
                'spans_label': spans_label,
                'entity_pairs': entity_pairs,
                'entity_pairs_label': entity_pairs_label
            })
            c_ner += Counter(list(refined_entities.values()))
            c_re += Counter(list(refined_relations.values()))
        n_ner = Counter([len(i['entities']) for i in self.data])
        n_re = Counter([len(i['relations']) for i in self.data])
        print("counted ner: ", sum(list(c_ner.values())))
        print("entity label stats:", dict(c_ner))
        print("relation label stats:", dict(c_re))
        print("span len stats:", dict(c_span_len))
        print("num of entity stats:", dict(n_ner))
        print("num of relation stats:", dict(n_re))
            # need stats
            # for i in 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class NERDataloader(DataLoader):
    def __init__(self, *args, **kargs):
        # DataLoader.__init__(*args, **kargs, collate_fn=ner_collate_fn)
        '''
        print(args, kargs)
        # super(__class__,self).__init__()
        super(__class__, self).__init__(*args, **kargs, collate_fn=ner_collate_fn)
        '''
        super(__class__, self).__init__(*args, **kargs, collate_fn=ner_collate_fn)
        

class REDataloader(DataLoader):
    def __init__(self, *args, **kargs):
        # DataLoader.__init__(*args, **kargs, collate_fn=ner_collate_fn)
        super(__class__, self).__init__(*args, **kargs, collate_fn=re_collate_fn)
        
class JREDataloader(DataLoader):
    def __init__(self, *args, **kargs):
        # DataLoader.__init__(*args, **kargs, collate_fn=ner_collate_fn)
        super(__class__, self).__init__(*args, **kargs, collate_fn=jre_collate_fn)