from transformers import BertForMaskedLM
from torch.nn import CrossEntropyLoss
from ToksTransformer import *
from data_utils import *
import torch.nn as nn
import pickle as pkl
import torch
import math
from pytorch_pretrained_bert import BertModel

law_cls = 103
accu_cls = 119
term_cls = 11


class CustomBertModel(BertForMaskedLM):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        if masked_lm_labels is None:
            return sequence_output
        prediction_scores = self.cls(sequence_output)
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        return masked_lm_loss, sequence_output


class CrossAttention(torch.nn.Module):
    def __init__(self, law_cls=law_cls, accu_cls=accu_cls, term_cls=11):
        super().__init__()
        self.laws = torch.from_numpy(pkl.load(open("laws_rep.pkl", "rb"))).cuda()
        self.law_masks = torch.from_numpy(pkl.load(open("laws_mask.pkl", "rb"))).unsqueeze(0).unsqueeze(2).cuda()
        self.law_embs = torch.from_numpy(pkl.load(open("laws_emb.pkl", "rb"))).cuda()
        self.law_k = nn.Linear(768, 768)
        self.fact_q = nn.Linear(768, 768)
        self.law_v = nn.Linear(768, 768)
        self.law_prob = nn.Linear(512, 1)
        self.d_prob = nn.Linear(768, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.accu_cls = nn.Linear(law_cls, accu_cls)
        self.term_cls = nn.Linear(law_cls, term_cls)

    def forward(self, fact_rep):
        facts_q = self.fact_q(fact_rep)
        laws_k = self.law_k(self.law_embs)
        laws_v = self.law_v(self.law_embs)
        relevance_scores = torch.einsum("fsd,lrd->flsr", facts_q, laws_k)
        relevance_scores = relevance_scores / math.sqrt(768)
        relevance_scores = relevance_scores + self.law_masks
        relevance_scores = nn.Softmax(dim=-1)(relevance_scores)
        relevance_scores = self.dropout(relevance_scores)
        laws = torch.matmul(relevance_scores, laws_v)
        laws = self.d_prob(laws).squeeze()
        law = self.law_prob(laws).squeeze()
        return law, self.accu_cls(law), self.term_cls(law)

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class BERTLegalCA(torch.nn.Module):
    def __init__(self, law_cls=law_cls, accu_cls=accu_cls, term_cls=11):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.laws = torch.from_numpy(pkl.load(open("laws_rep.pkl", "rb"))).cuda()
        self.law_q = nn.Linear(768, 768)
        self.fact_k = nn.Linear(768, 768)
        self.fact_v = nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.law_d = nn.Linear(768, 1)
        self.accu_d = nn.Linear(768, 1)
        self.term_d = nn.Linear(768, 1)
        self.law1 = nn.Linear(512, 1)
        self.accu1 = nn.Linear(512, 1)
        self.term1 = nn.Linear(512, 1)
        self.law2 = nn.Linear(law_cls, law_cls)
        self.accu2 = nn.Linear(law_cls, accu_cls)
        self.term2 = nn.Linear(law_cls, term_cls)
        #         self.emb_layer = nn.Linear(768, 768)
        #         self.attn_layer = nn.Linear(768, 1)
        self.act = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels=None):
        if masked_lm_labels is not None:
            mlm_loss, output = self.bert(input_ids, token_type_ids, attention_mask, masked_lm_labels)
        else:
            output = self.bert(input_ids, token_type_ids, attention_mask)
        laws_q = self.law_q(self.laws)
        facts_k = self.fact_k(output)
        facts_v = self.fact_v(output)
        relevance_scores = torch.einsum("lrd,fsd->lfrs", laws_q, facts_k)  # 101,4,512(law),512(fact)
        relevance_scores = relevance_scores / math.sqrt(768)
        relevance_scores = relevance_scores + (1.0 - attention_mask.unsqueeze(0).unsqueeze(2)) * -10000.0
        relevance_scores = nn.Softmax(dim=-1)(relevance_scores)
        relevance_scores = self.dropout(relevance_scores)
        a_facts = torch.matmul(relevance_scores, facts_v)  # 101,4,512(fact),768
        g = a_facts.permute(1, 0, 2, 3).contiguous()  # 4,101,512(fact),768
        #         att = torch.sigmoid(self.attn_layer(a_facts))  # 4,101,512,1
        #         emb = self.act(self.emb_layer(a_facts))  # 4,101,512,768
        #         g = attention_mask.unsqueeze(1).unsqueeze(-1) * a_facts  # 4,101,512(fact),768
        #         g = torch.sum(g, dim=-2) / attention_mask.sum(dim=1).unsqueeze(1).unsqueeze(-1) \
        #             + torch.max(g, dim=-2)[0]  # 4,101,768
        #         g = self.dropout(g)
        if masked_lm_labels is not None:
            return self.law1(self.act(self.law_d(g).squeeze(-1))).squeeze(-1), \
                   self.accu2(self.act(self.accu1(self.act(self.accu_d(g).squeeze(-1))).squeeze(-1))), \
                   self.term2(self.act(self.term1(self.act(self.term_d(g).squeeze(-1))).squeeze(-1))), \
                   mlm_loss
        else:
            return self.law1(self.act(self.law_d(g).squeeze(-1))).squeeze(-1), \
                   self.accu2(self.act(self.accu1(self.act(self.accu_d(g).squeeze(-1))).squeeze(-1))), \
                   self.term2(self.act(self.term1(self.act(self.term_d(g).squeeze(-1))).squeeze(-1)))

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class BERTLegalWithLawEmb(torch.nn.Module):
    def __init__(self, law_cls=law_cls, accu_cls=accu_cls, term_cls=11):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.ruledataset = LegalRuleDataset()
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.laws = torch.from_numpy(pkl.load(open("laws.pkl", "rb")))
        self.law_masks = torch.from_numpy(pkl.load(open("law_masks.pkl", "rb")))
        self.law_embs = torch.from_numpy(pkl.load(open("law_embs.pkl", "rb")))
        self.law_k = nn.Linear(768, 768)
        self.fact_q = nn.Linear(768, 768)
        self.law_v = nn.Linear(768, 768)
        self.law_prob = nn.Linear(512, 1)
        self.d_prob = nn.Linear(768, 1)
        self.fact_law = nn.Linear(law_cls, law_cls)
        self.rule_law = nn.Linear(law_cls, law_cls)
        self.dropout = torch.nn.Dropout(0.1)
        self.law = nn.Linear(768, law_cls)
        self.accu = nn.Linear(768, accu_cls)
        self.term = nn.Linear(768, term_cls)

    def forward(self, input_ids, token_type_ids, attention_mask, plm=False):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        law = self.law(output[:, 0, :])
        facts_q = self.fact_q(self.bert.bert.embeddings(input_ids))
        laws_k = self.law_k(self.law_embs)
        laws_v = self.law_v(self.law_embs)
        relevance_scores = torch.einsum("fsd,lrd->flsr", facts_q, laws_k)
        relevance_scores = relevance_scores / math.sqrt(768)
        relevance_scores = relevance_scores + self.law_masks
        relevance_scores = nn.Softmax(dim=-1)(relevance_scores)
        relevance_scores = self.dropout(relevance_scores)
        laws = torch.matmul(relevance_scores, laws_v)
        laws = self.d_prob(laws).squeeze()
        scores = self.law_prob(laws).squeeze()
        law = self.rule_law(law) + self.fact_law(scores)
        law_no = torch.argmax(law, dim=1)
        if plm:
            law_batch = [self.ruledataset[ln] for ln in law_no]
            record = collate_fn_law(law_batch)
            mlm_loss, law_out = self.bert(record[0], record[1], record[2], record[3])
            self.laws[law_no] = law_out[:, 0, :]
            law_language_pred = self.law(self.dropout(law_out[:, 0, :]))
            # output = torch.cat([output[:, 0, :], law_out[:, 0, :]], dim=1)
        else:
            law_batch = self.laws[law_no]
            law_language_pred = self.law(self.dropout(law_batch))
            mlm_loss = 0.
            # output = torch.cat([output[:, 0, :], law_batch], dim=1)
        output = output[:, 0, :]
        return law, self.accu(self.dropout(output)), \
               self.term(self.dropout(output)), law_language_pred, mlm_loss, law_no.long()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)
        pkl.dump(self.laws.cpu().data.numpy(), open("laws.pkl", "wb"))

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        self.laws = torch.from_numpy(pkl.load(open("laws.pkl", "rb"))).cuda()


class BERTLegal(torch.nn.Module):
    def __init__(self, law_cls=law_cls, accu_cls=accu_cls, term_cls=11):
        super().__init__()
        import os
        self.BERT_MODEL = 'bert-base-chinese'
        self.ruledataset = LegalRuleDataset()
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        if os.path.exists("laws.pkl"):
            self.laws = torch.from_numpy(pkl.load(open("laws.pkl", "rb"))).cuda()
        else:
            self.laws = torch.ones((len(self.ruledataset), 512, 768)).cuda()
        self.dropout = torch.nn.Dropout(0.2)
        self.law_d = nn.Linear(768, 1)
        self.accu_d = nn.Linear(768, 1)
        self.term_d = nn.Linear(768, 1)
        self.law1 = nn.Linear(512, law_cls)
        self.accu1 = nn.Linear(512, accu_cls)
        self.term1 = nn.Linear(512, term_cls)
        self.act = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, plm=False):
        output = self.bert(input_ids, token_type_ids, attention_mask)  # 4, 512, 768
        law = self.law1(self.act(self.law_d(self.dropout(output)).squeeze(-1)))
        law_no = torch.argmax(law, dim=1)
        if plm:
            law_batch = [self.ruledataset[ln] for ln in law_no]
            record = collate_fn_law(law_batch)
            mlm_loss, law_out = self.bert(record[0], record[1], record[2], record[3])
            self.laws[law_no] = law_out
            law_language_pred = self.law1(self.act(self.law_d(self.dropout(law_out)).squeeze(-1)))
            # output = torch.cat([output[:, 0, :], law_out[:, 0, :]], dim=1)
        else:
            law_batch = self.laws[law_no]
            law_language_pred = self.law1(self.act(self.law_d(self.dropout(law_batch)).squeeze(-1)))
            mlm_loss = 0.
            # output = torch.cat([output[:, 0, :], law_batch], dim=1)
        return law, self.accu1(self.act(self.accu_d(self.dropout(output)).squeeze(-1))), \
               self.term1(
                   self.act(self.term_d(self.dropout(output)).squeeze(-1))), law_language_pred, mlm_loss, law_no.long()

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)
        pkl.dump(self.laws.cpu().data.numpy(), open("laws.pkl", "wb"))

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        # self.laws = torch.from_numpy(pkl.load(open("laws.pkl", "rb"))).cuda()


class BERTLegalFinetune(torch.nn.Module):
    def __init__(self, law_cls=law_cls, accu_cls=accu_cls, term_cls=11):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = nn.Linear(768, law_cls)
        self.accu_cls = nn.Linear(768, accu_cls)
        self.term_cls = nn.Linear(768, term_cls)
        self.qa_outputs = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        return self.cls(self.dropout(output[:, 0, :])), \
               self.accu_cls(self.dropout(output[:, 0, :])), \
               self.term_cls(self.dropout(output[:, 0, :]))

    def get_span(self, input_ids, token_type_ids, attention_mask):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return torch.argmax(start_logits), torch.argmax(end_logits)

    def stage_1(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, labels,
                start_positions, end_positions):
        mlm_loss, output = self.bert(input_ids, token_type_ids, attention_mask, masked_lm_labels)
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        span_loss = (start_loss + end_loss) / 2
        loss = loss_fct(self.cls(self.dropout(output[:, 0, :])), labels)
        return loss, mlm_loss, span_loss

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class BERTLegalPretraining(torch.nn.Module):
    def __init__(self, cls_num):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.qa_outputs = nn.Linear(768, 2)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = nn.Linear(768, cls_num)

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels,
                start_positions, end_positions):
        mlm_loss, output = self.bert(input_ids, token_type_ids, attention_mask, masked_lm_labels)
        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        span_loss = (start_loss + end_loss) / 2

        return self.cls(self.dropout(output[:, 0, :])), mlm_loss, span_loss

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class BERTLJP(nn.Module):
    def __init__(self, input_fact, with_distill, mask_stategy, multi_head=True):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        self.toks_transformer = ToksTransformer(withDistill=with_distill,
                                                num_hidden_layers=3,
                                                strategy=mask_stategy,
                                                input_fact=input_fact,
                                                multi_head=multi_head)
        self.dropout = torch.nn.Dropout(0.1)
        self.law_w = torch.nn.Linear(768, 768)
        self.accu_w = torch.nn.Linear(768, 768)
        self.term_w = torch.nn.Linear(768, 768)
        self.fact_w = torch.nn.Linear(768, 768)
        self.law_pred = torch.nn.Linear(768, 1)
        self.accu_pred = torch.nn.Linear(768, 1)
        self.term_pred = torch.nn.Linear(768, 1)
        self.act = nn.Tanh()
        self.input_fact = input_fact

    def forward(self, input_ids, token_type_ids, attention_mask):
        if not self.input_fact:
            toks_emb = self.toks_transformer(hidden_states=None)
            output = self.bert(input_ids, token_type_ids, attention_mask)
            law_emb = self.act(self.law_w(toks_emb[0, :law_cls, :]))
            accu_emb = self.act(self.accu_w(toks_emb[0, law_cls:law_cls + accu_cls, :]))
            term_emb = self.act(self.term_w(toks_emb[0, law_cls + accu_cls:law_cls + accu_cls + term_cls, :]))
            fact_emb = self.act(self.fact_w(output[:, 0, :]))
            law_pred = torch.matmul(fact_emb, law_emb.T)
            accu_pred = torch.matmul(fact_emb, accu_emb.T)
            term_pred = torch.matmul(fact_emb, term_emb.T)
            return law_pred, accu_pred, term_pred
        else:
            output = self.bert(input_ids, token_type_ids, attention_mask)
            toks_emb = self.toks_transformer(hidden_states=output[:, 0, :])
            law_emb = self.act(self.law_w(toks_emb[:, :law_cls, :]))  # 4,law_cls,768
            accu_emb = self.act(self.accu_w(toks_emb[:, law_cls:law_cls + accu_cls, :]))
            term_emb = self.act(self.term_w(toks_emb[:, law_cls + accu_cls:law_cls + accu_cls + term_cls, :]))
            law_pred = self.law_pred(law_emb).squeeze(-1)
            accu_pred = self.accu_pred(accu_emb).squeeze(-1)
            term_pred = self.term_pred(term_emb).squeeze(-1)
            return law_pred, accu_pred, term_pred

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)


class BERTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.BERT_MODEL = 'bert-base-chinese'
        self.bert = CustomBertModel.from_pretrained(self.BERT_MODEL)
        law2accu = pkl.load(open('law2accu.pkl', 'rb'))
        law2term = pkl.load(open('law2term.pkl', 'rb'))
        accu2term = pkl.load(open("accu2term.pkl", "rb"))
        adj = np.eye(233)
        law_cls = 103
        accu_cls = 119
        term_cls = 11
        adj[:law_cls, law_cls:law_cls + accu_cls] = law2accu
        adj[:law_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = law2term
        adj[:, :law_cls] = adj[:law_cls].T
        adj[law_cls:law_cls + accu_cls, law_cls + accu_cls:law_cls + accu_cls + term_cls] = accu2term
        adj[law_cls + accu_cls:law_cls + accu_cls + term_cls, law_cls:law_cls + accu_cls] = accu2term.T
        adj = normalize_adj(adj)
        self.GNN_q = GNN(in_dim=768, out_dim=1, adj=torch.from_numpy(adj).to(torch.float32))
        self.toks_embeddings = BertEmbeddings()
        self.token_ids = torch.range(0, 232).unsqueeze(0).long().cuda()
        self.token_type_ids = torch.cat([torch.ones((1, law_cls)) * 0,
                                         torch.ones((1, accu_cls)),
                                         torch.ones((1, term_cls)) * 2],
                                        dim=1).long().cuda()

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids, token_type_ids, attention_mask)
        tok_emb = self.toks_embeddings(self.token_ids, self.token_type_ids)
        tok_emb = tok_emb.expand([len(input_ids), law_cls + term_cls + accu_cls, 768])
        hidden_states = output[:, 0, :]  # 4, 768
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = hidden_states.expand([len(input_ids), law_cls + term_cls + accu_cls, 768])
        gnn_inputs = hidden_states + tok_emb
        gnn_outputs = self.GNN_q(gnn_inputs).squeeze(-1)  # 4,233,1
        law_pred = gnn_outputs[:, :law_cls]
        accu_pred = gnn_outputs[:, law_cls:law_cls + accu_cls]
        term_pred = gnn_outputs[:, law_cls + accu_cls:law_cls + accu_cls + term_cls]
        return law_pred, accu_pred, term_pred

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
