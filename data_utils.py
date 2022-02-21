from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from glob import glob
import pickle as pkl
import numpy as np
import random
import torch
import json
import re


def send_email(content):
    import smtplib
    from email.mime.text import MIMEText
    msg_from = ''
    passwd = ''
    msg_to = ''
    subject = "log信息"
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        s = smtplib.SMTP_SSL("smtp.qq.com", 465)
        s.login(msg_from, passwd)
        s.sendmail(msg_from, msg_to, msg.as_string())
        # s.sendmail(msg_from, "284467290@qq.com", msg.as_string())
    except:
        pass


def gen_mp():
    dataset = CrimeFactDataset("train")
    law_mp_accu = {}
    law_mp_term = {}
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        if sample['law'] not in law_mp_accu:
            law_mp_accu[sample['law']] = [0] * 117
        law_mp_accu[sample['law']][sample['accu']] += 1
        if sample['law'] not in law_mp_term:
            law_mp_term[sample['law']] = [0] * 11
        law_mp_term[sample['law']][sample['term']] += 1
    pkl.dump(law_mp_term, open("law_mp_term.pkl", "wb"))
    pkl.dump(law_mp_accu, open("law_mp_term.pkl", "wb"))


def collate_fn_law(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    label = torch.stack([x['label'] for x in batch]).squeeze()
    return input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), masked_lm_labels.cuda(), label.cuda().long()


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    label = [x['label'] for x in batch]
    label = torch.from_numpy(np.array(label))
    start = [x['start'] for x in batch]
    start = torch.from_numpy(np.array(start))
    end = [x['end'] for x in batch]
    end = torch.from_numpy(np.array(end))
    return input_ids.cuda(), \
           token_type_ids.cuda(), \
           attention_mask.cuda(), \
           masked_lm_labels.cuda(), \
           label.cuda().long(), \
           start.cuda().long(), \
           end.cuda().long()


def collate_fn_fact_pkl(batch):
    rep = torch.stack([torch.from_numpy(x['rep']) for x in batch]).squeeze(1)
    emb = torch.stack([torch.from_numpy(x['emb']) for x in batch]).squeeze(1)
    mask = torch.stack([torch.from_numpy(x['mask']) for x in batch]).squeeze(1)
    accu = [x['accu'] for x in batch]
    accu = torch.from_numpy(np.array(accu))
    law = [x['law'] for x in batch]
    law = torch.from_numpy(np.array(law))
    term = [x['term'] for x in batch]
    term = torch.from_numpy(np.array(term))
    return rep.cuda(), emb.cuda(), mask.cuda(), law.cuda().long(), accu.cuda().long(), term.cuda().long()


def collate_fn_fact(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    accu = [x['accu'] for x in batch]
    accu = torch.from_numpy(np.array(accu))
    law = [x['law'] for x in batch]
    law = torch.from_numpy(np.array(law))
    term = [x['term'] for x in batch]
    term = torch.from_numpy(np.array(term))
    return input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), \
           law.cuda().long(), accu.cuda().long(), term.cuda().long(), \
           masked_lm_labels.cuda()


def collate_fn_gnn_p(batch):
    inputs = [x['inputs'] for x in batch]
    inputs = torch.from_numpy(np.array(inputs))
    targets = [x['targets'] for x in batch]
    targets = torch.from_numpy(np.array(targets))
    return inputs.unsqueeze(-1).cuda(), targets.cuda()


class CrimeFactDatasetPKL(Dataset):
    def __init__(self, mode):
        self.fact_list = glob("../dataset/%s_cs/*.pkl" % mode)

    def __getitem__(self, item):
        return pkl.load(open(self.fact_list[item], "rb"))

    def __len__(self):
        return len(self.fact_list)


class CrimeFactDatasetUniform(Dataset):
    def __init__(self, mode):
        fact_list = open("%s_cs.json" % mode, 'r', encoding='UTF-8').readlines()
        delete_index = []
        for i, sample in enumerate(fact_list):
            sample = json.loads(sample)
            if len(sample['fact_cut']) <= 50:
                delete_index.append(i)
        delete_index = delete_index[::-1]
        for idx in delete_index:
            fact_list.pop(idx)
        self.fact_list = fact_list
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = 512
        self.law_mp_index = pkl.load(open("law_mp_index.pkl", "rb"))

    def __getitem__(self, item):
        item = item % 101
        item = random.choice(self.law_mp_index[item])
        sample = self.fact_list[item]
        sample = json.loads(sample)
        fact = sample['fact_cut'].replace(" ", "")
        if len(fact) > 510:
            fact = fact[:255] + fact[-255:]
        ret = self.tokenizer(fact, max_length=512, padding="max_length", return_tensors="pt")
        ret['accu'] = sample['accu']
        ret['law'] = sample['law']
        ret['term'] = sample['term']
        return ret

    def __len__(self):
        return 10000


class GNNpDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        ret = {'inputs': self.inputs[item], 'targets': self.targets[item]}
        return ret

    def __len__(self):
        return len(self.inputs)


class CrimeFactDataset(Dataset):
    def __init__(self, mode, mask_rate=0.):
        fact_list = open("%s_cs.json" % mode, 'r', encoding='UTF-8').readlines()
        delete_index = []
        for i, sample in enumerate(fact_list):
            sample = json.loads(sample)
            if len(sample['fact_cut']) <= 10:
                delete_index.append(i)
        delete_index = delete_index[::-1]
        for idx in delete_index:
            fact_list.pop(idx)
        self.fact_list = fact_list
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = 512
        self.mask_rate = mask_rate

    def __getitem__(self, item):
        sample = self.fact_list[item]
        sample = json.loads(sample)
        fact = sample['fact_cut'].replace(" ", "")
        if len(fact) > 510:
            fact = fact[:255] + fact[-255:]
        ret = self.tokenizer(fact, max_length=512, padding="max_length", return_tensors="pt")
        fact_tok = self.tokenizer.tokenize(fact)
        ret['masked_lm_labels'] = ret['input_ids'].clone()
        length = int(torch.sum(ret['attention_mask']))
        ret['masked_lm_labels'][0][0] = ret['masked_lm_labels'][0][length - 1] = -1
        ret['input_ids'][0][random.choices(range(1, length), k=int((length - 2) * self.mask_rate))] = 103
        ret['accu'] = sample['accu']
        ret['law'] = sample['law']
        ret['term'] = sample['term']
        ret['fact_tok'] = fact_tok
        return ret

    def __len__(self):
        return len(self.fact_list)


class LegalRuleDataset(Dataset):
    def __init__(self, mask_rate=0.15):
        print("mask_rate", mask_rate)
        self.mask_rate = mask_rate
        rule_list = open("new_law.txt").readlines()
        self.rule_list = [int(rule.strip()) for rule in rule_list]
        rule = open("rulenew.txt", 'r', encoding='UTF-8').readlines()
        rule = [line.strip().split("\t")[1].replace("之一\u3000", "").replace("\u3000", "") for line in rule]
        rule = [re.sub(u"\\(.*?\\)|\\{.*?\\}|\\【.*?\\】|\\<.*?\\>", "", s) for s in rule]
        self.rule = rule
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = 512
        self.start_end = pkl.load(open("start_end.pkl", "rb"))

    def __getitem__(self, item):
        rule_no = self.rule_list[item] - 1
        ret = self.tokenizer(self.rule[rule_no], max_length=512, padding="max_length", return_tensors="pt")
        ret['masked_lm_labels'] = -torch.ones((1, 512)).long()
        length = int(torch.sum(ret['attention_mask']))
        random_list = random.choices(range(1, length), k=int((int(length - 2) * self.mask_rate)))
        ret['masked_lm_labels'][:, random_list] = ret['input_ids'][:, random_list]
        ret['input_ids'][:, random_list[:int(0.8 * len(random_list))]] = 103
        if int(0.1 * len(random_list)):
            ret['input_ids'][:, random_list[-int(0.1 * len(random_list)):]] = torch.from_numpy(
                np.array(random.choices(range(self.tokenizer.vocab_size), k=int(0.1 * len(random_list)))))
        ret.update({"rule_no": self.rule_list[item], "rule_text": self.rule[rule_no], "label": item})
        ret['start'] = int(self.start_end[self.rule_list[item]][0])
        ret['end'] = int(self.start_end[self.rule_list[item]][1])
        return ret

    def __len__(self):
        return len(self.rule_list)
