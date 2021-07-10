from dataset import *
from model import *
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AlbertTokenizer, AlbertModel
import argparse
from tqdm import *

class Framework():
    def __init__(self, args):
        self.args = args
        if args.plm == 'DistilBert':
            encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            hid_dim = 768
        elif args.plm == 'Bert':
            encoder = BertModel.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            hid_dim = 768
        elif args.plm == 'Roberta':
            encoder = RobertaModel.from_pretrained('roberta-base')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            hid_dim = 768
        elif args.plm == 'Albert':
            encoder = AlbertModel.from_pretrained('albert-xlarge-v2')
            tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
            hid_dim = 2048
        else:
            raise 'Unimplemented'
        
        self.testset = JREDataset(args.data_path + '/test.ACE05.json', tokenizer, 8)
        self.validset = JREDataset(args.data_path + "/valid.ACE05.json", tokenizer, 8)
        self.trainset = JREDataset(args.data_path + "/train.ACE05.json", tokenizer, 8)

        self.test_ner_loader = NERDataloader(self.testset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.valid_ner_loader = NERDataloader(self.validset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.train_ner_loader = NERDataloader(self.trainset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.test_re_loader = REDataloader(self.testset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.valid_re_loader = REDataloader(self.validset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.train_re_loader = REDataloader(self.trainset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
    
        self.ner_model = BertSpanNER(encoder, hid_dim)
        self.re_model = BertRE(encoder, hid_dim)
        self.ner_log_path = 'ner_log/' + args.model_name + '.log'
        self.re_log_path = 're_log/' + args.model_name + '.log'
        self.log_path = ''
        
    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(self.log_path, 'a+') as f_log:
                f_log.write(s + '\n')
        
    def evaluateNER(self, ner_loader, n_total_ner):
        print("evaluating...")
        self.log_path = self.ner_log_path
        # c_time = time.time()
        a_cor, a_tot = 0, n_total_ner
        a_pre = 0
        l_cor = 0
        l_tot = 0
        l_pred = 0
        # l_total_cand = 0
        self.ner_model.eval()
        for l in tqdm(ner_loader):
            seq, mask, span_batch, span_label_batch, span_index = l
            with torch.no_grad():
                res = self.ner_model(seq, mask, span_batch, span_index)
                l_tot += torch.sum(span_label_batch != 0).item()
                a_tot += span_label_batch.shape[0]
                predicted = torch.argmax(res, dim=1)
                l_pred += torch.sum(predicted != 0).item()
                a_cor += torch.sum(span_label_batch == predicted).item()
                l_cor += torch.sum((span_label_batch == predicted)*(span_label_batch != 0)).item()
            
        acc = a_cor / a_tot
        self.logging('all accuracy: %4f'%acc)
        self.logging('for valid spans: cor: %d, pred: %d, tot: %d, cand tot: %d'%(l_cor, l_pred, n_total_ner, l_tot))
        p = l_cor / l_pred
        r = l_cor / n_total_ner
        f1 = 2 * (p * r) / (p + r + 1e-6)
        self.logging('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
        
        self.ner_model.train()
        return f1
    
    def evaluateRE(self, re_loader, n_total_re):
        print("evaluating...")
        self.log_path = self.re_log_path
        # c_time = time.time()
        a_cor, a_tot = 0, n_total_re
        a_pre = 0
        l_cor = 0
        l_tot = 0
        l_pred = 0
        # l_total_cand = 0
        self.re_model.eval()
        for l in tqdm(re_loader):
            # seq, mask, span_batch, span_label_batch, span_index = l
            seq, mask, pair_batch, pair_label_batch, pair_index = l
            with torch.no_grad():
                res = self.re_model(seq, mask, pair_batch, pair_index)
                l_tot += torch.sum(pair_label_batch != 0).item()
                a_tot += pair_label_batch.shape[0]
                predicted = torch.argmax(res, dim=1)
                l_pred += torch.sum(predicted != 0).item()
                a_cor += torch.sum(pair_label_batch == predicted).item()
                l_cor += torch.sum((pair_label_batch == predicted)*(pair_label_batch != 0)).item()
            
        acc = a_cor / a_tot
        self.logging('all accuracy: %4f'%acc)
        self.logging('for valid spans: cor: %d, pred: %d, tot: %d, cand tot: %d'%(l_cor, l_pred, n_total_re, l_tot))
        p = l_cor / (l_pred + 1e-6)
        r = l_cor / n_total_re
        f1 = 2 * (p * r) / (p + r + 1e-6)
        self.logging('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
        
        self.re_model.train()
        return f1

    def trainNER(self):
        best_f1 = 0
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.ner_model.parameters(), lr=self.args.lr)
        self.ner_model.to('cuda')
        for epoch in range(self.args.epoch):
            # break
            for step, i in tqdm(enumerate(self.train_re_loader)):
                # break
                seq, mask, span_batch, span_label_batch, span_index = i
                res = self.ner_model(seq, mask, span_batch, span_index)
                loss = criterion(res, span_label_batch)    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.ner_model.zero_grad()
                # print(loss)
                # bs = len(span_label_batch)
                '''
                if step % 10 == 0:
                    print(loss)
                    valid = torch.sum(span_label_batch != 0)
                    predicted = torch.argmax(res, dim=1)
                    true_valid = torch.sum((span_label_batch == predicted)*(span_label_batch != 0))
                # print(res, span_label_batch)
                    print(true_valid/valid, true_valid, valid)
                '''
            f1 = self.evaluateNER(self.valid_ner_loader, self.validset.c_ori_ner)
            self.logging("epoch %d f1: %4f"%(epoch, f1))
            if f1 > best_f1:
                state = self.ner_model.state_dict()
                torch.save(state, "saved_ner_models/" + self.args.model_name + '.ckpt')
                
        checkpoint = torch.load("saved_ner_models/" + self.args.model_name + '.ckpt')
        self.ner_model.load_state_dict(checkpoint)
        print("on test set:", self.evaluateNER(self.test_ner_loader, self.testset.c_ori_ner))

    def trainRE(self):
        best_f1 = 0
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.re_model.parameters(), lr=self.args.lr)
        self.re_model.to('cuda')
        for epoch in range(self.args.epoch):
            # break
            for step, i in tqdm(enumerate(self.train_re_loader)):
                # break
                seq, mask, pair_batch, pair_label_batch, pair_index = i
                # seq, mask, span_batch, span_label_batch, span_index = i
                res = self.re_model(seq, mask, pair_batch, pair_index)
                loss = criterion(res, pair_label_batch)    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.re_model.zero_grad()
                # print(loss)
                # bs = len(span_label_batch)
                '''
                if step % 10 == 0:
                    print(loss)
                    valid = torch.sum(span_label_batch != 0)
                    predicted = torch.argmax(res, dim=1)
                    true_valid = torch.sum((span_label_batch == predicted)*(span_label_batch != 0))
                # print(res, span_label_batch)
                    print(true_valid/valid, true_valid, valid)
                '''
            f1 = self.evaluateRE(self.valid_re_loader, self.validset.c_ori_re)
            self.logging("epoch %d f1: %4f"%(epoch, f1))
            if f1 > best_f1:
                state = self.re_model.state_dict()
                torch.save(state, "saved_re_models/" + self.args.model_name + '.ckpt')
                
        checkpoint = torch.load("saved_re_models/" + self.args.model_name + '.ckpt')
        self.re_model.load_state_dict(checkpoint)
        print("on test set:", self.evaluateRE(self.test_re_loader, self.testset.c_ori_re))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plm", type=str, default='DistilBert')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--model_name", type=str, default='unnamed')
    args = parser.parse_args()
    f = Framework(args)
    f.trainRE()
    # f.trainNER()