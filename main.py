from dataset import *
from model import *
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AlbertTokenizer, AlbertModel
import transformers
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
            assert False, 'Unimplemented'
        
        self.testset = JREDataset(args.data_path + '/test.ACE05.json', tokenizer, 8)
        
        self.test_ner_loader = NERDataloader(self.testset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        self.test_re_loader = REDataloader(self.testset, batch_size=args.bs, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0)
        if args.task != 'jre':
        
            self.validset = JREDataset(args.data_path + "/valid.ACE05.json", tokenizer, 8)
            self.trainset = JREDataset(args.data_path + "/train.ACE05.json", tokenizer, 8)

            
            self.valid_ner_loader = NERDataloader(self.validset, batch_size=args.bs, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0)
            self.train_ner_loader = NERDataloader(self.trainset, batch_size=args.bs, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0)
            self.valid_re_loader = REDataloader(self.validset, batch_size=args.bs, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0)
            self.train_re_loader = REDataloader(self.trainset, batch_size=args.bs, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0)
    
        self.ner_model = BertSpanNER(encoder, hid_dim)
        self.re_model = BertRE(encoder, hid_dim, args.attention)
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
        p = l_cor / (l_pred + 1e-6)
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
        param_optimizer = list(self.ner_model.named_parameters())
        print("param name", [n for n, p in param_optimizer])
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'encoder' in n]},
            {'params': [p for n, p in param_optimizer
                if 'encoder' not in n], 'lr': args.tlr}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        t_total = len(self.trainset) // self.args.bs * self.args.epoch
        if self.args.warmup_proportion > 0:
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(t_total*self.args.warmup_proportion), t_total)
        
        # optimizer = torch.optim.Adam(self.ner_model.parameters(), lr=self.args.lr)
        self.ner_model.to('cuda')
        total_step = 0
        for epoch in range(self.args.epoch):
            # break
            for step, i in tqdm(enumerate(self.train_ner_loader)):
                # break
                total_step += 1
                seq, mask, span_batch, span_label_batch, span_index = i
                res = self.ner_model(seq, mask, span_batch, span_index)
                loss = criterion(res, span_label_batch)    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.ner_model.zero_grad()
                
                if self.args.warmup_proportion > 0:
                    scheduler.step()
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
            f1_t = self.evaluateNER(self.test_ner_loader, self.testset.c_ori_ner)
            self.logging("epoch %d f1: %4f, f1_test: %4f"%(epoch, f1, f1_t))
            if f1 > best_f1:
                best_f1 = f1
                # state = self.ner_model.state_dict()
                torch.save(self.ner_model, "saved_ner_models/" + self.args.model_name + '.ckpt')
                
        # checkpoint = torch.load("saved_ner_models/" + self.args.model_name + '.ckpt')
        # self.ner_model.load_state_dict(checkpoint)
        self.ner_model = torch.load("saved_ner_models/" + self.args.model_name + '.ckpt')
        print("on test set:", self.evaluateNER(self.test_ner_loader, self.testset.c_ori_ner))

    def trainRE(self):
        best_f1 = 0
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.re_model.parameters(), lr=self.args.lr)
        t_total = len(self.trainset) // self.args.bs * self.args.epoch
        
        if self.args.warmup_proportion > 0:
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup_proportion), t_total)
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
                
                if self.args.warmup_proportion > 0:
                    scheduler.step()
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
            f1_t = self.evaluateRE(self.test_re_loader, self.testset.c_ori_re)
            self.logging("epoch %d f1: %4f, f1_t: %4f"%(epoch, f1, f1_t))
            if f1 > best_f1:
                # state = self.re_model.state_dict()
                best_f1 = f1
                torch.save(self.re_model, "saved_re_models/" + self.args.model_name + '.ckpt')
                
        self.re_model = torch.load("saved_re_models/" + self.args.model_name + '.ckpt')
        # self.re_model.load_state_dict(checkpoint)
        print("on test set:", self.evaluateRE(self.test_re_loader, self.testset.c_ori_re))

    def testJRE(self):
        if self.args.re_model_name == 'none' or self.args.ner_model_name == 'none':
            assert False, 'no specified trained models'
        self.re_model = torch.load("saved_re_models/" + self.args.re_model_name + '.ckpt')
        self.ner_model = torch.load("saved_ner_models/" + self.args.ner_model_name + '.ckpt')
        
        print("evaluating...")
        self.log_path = self.re_log_path
        # c_time = time.time()
        pred_ls = []
        label_ls = []
        # l_total_cand = 0
        self.re_model.eval()
        for l in tqdm(self.test_ner_loader):
            # seq, mask, span_batch, span_label_batch, span_index = l
            seq, mask, pair_batch, pair_label_batch, pair_index = l
            with torch.no_grad():
                res = self.ner_model(seq, mask, pair_batch, pair_index)
                predicted = torch.argmax(res, dim=1)
                pred_ls.append(predicted)
                label_ls.append(pair_label_batch)
                # l_pred += torch.sum(predicted != 0).item()
                # a_cor += torch.sum(pair_label_batch == predicted).item()
                # l_cor += torch.sum((pair_label_batch == predicted)*(pair_label_batch != 0)).item()
        pred_ls = torch.cat(pred_ls, dim=0)
        label_ls = torch.cat(label_ls, dim=0)
        
        for i in range(1, 8):
            print(torch.sum((pred_ls == i) * (label_ls == i)))  # tp
            print(torch.sum((pred_ls == i)))  # predicted
            # labels are in the stats of dataset
        
        
        print("evaluating...")
        self.log_path = self.re_log_path
        # c_time = time.time()
        pred_ls = []
        label_ls = []
        # l_total_cand = 0
        self.re_model.eval()
        for l in tqdm(self.test_re_loader):
            # seq, mask, span_batch, span_label_batch, span_index = l
            seq, mask, pair_batch, pair_label_batch, pair_index = l
            with torch.no_grad():
                res = self.re_model(seq, mask, pair_batch, pair_index)
                predicted = torch.argmax(res, dim=1)
                pred_ls.append(predicted)
                label_ls.append(pair_label_batch)
                # l_pred += torch.sum(predicted != 0).item()
                # a_cor += torch.sum(pair_label_batch == predicted).item()
                # l_cor += torch.sum((pair_label_batch == predicted)*(pair_label_batch != 0)).item()
        pred_ls = torch.cat(pred_ls, dim=0)
        label_ls = torch.cat(label_ls, dim=0)
        for i in range(1, 7):
            print(torch.sum((pred_ls == i) * (label_ls == i)))  # tp
            print(torch.sum((pred_ls == i)))  # predicted
            # labels are in the stats of dataset
        
        
        # return 0
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='ner')
    parser.add_argument("--plm", type=str, default='Bert')
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tlr", type=float, default=1e-4)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--attention", action='store_true')
    parser.add_argument("--model_name", type=str, default='unnamed')
    parser.add_argument("--re_model_name", type=str, default='backup')
    parser.add_argument("--ner_model_name", type=str, default='ace05')
    args = parser.parse_args()
    f = Framework(args)
    if args.task == 'ner':
        f.trainNER()
    elif args.task == 're':
        f.trainRE()
    elif args.task == 'jre':
        f.testJRE()
    # f.trainNER()