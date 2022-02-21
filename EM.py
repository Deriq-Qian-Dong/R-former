from sklearn import metrics
from modeling import *
from tqdm import tqdm
import pickle as pkl
import torch.nn
import torch

law_cls = 103
accu_cls = 119
term_cls = 11


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_report(labels, preds):
    N_CLASSES = max(labels) + 1
    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))
    c = (preds == labels)
    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
    report = ""
    for i in range(N_CLASSES):
        if class_total[i]:
            report += 'Accuracy of %d : %d/%d=%.4f' % (
                i, class_correct[i], class_total[i], class_correct[i] / class_total[i]) + "\n"
    return report


def train_iteration(model, gnn_p, dataloader, soft_threshold=0.1):
    total_loss = 0
    accu_acc = 0
    law_acc = 0
    term_acc = 0
    total_law_loss = 0
    total_accu_loss = 0
    total_term_loss = 0
    total_soft_loss = 0

    law_preds = None
    term_preds = None
    accu_preds = None
    law_labels = np.array([])
    term_labels = np.array([])
    accu_labels = np.array([])

    model.train()
    gnn_p.eval()
    total = 0
    for step, record in enumerate(dataloader):
        law, accu, term = model(record[0], record[1], record[2])  # 喂给 net 训练数据 x, 输出分析值
        with torch.no_grad():
            law_p, accu_p, term_p = gnn_p(
                torch.cat([nn.Softmax(dim=-1)(law), nn.Softmax(dim=-1)(accu), nn.Softmax(dim=-1)(term)],
                          dim=1).unsqueeze(-1))
        _, soft_targets = get_inputs(nn.Softmax(dim=-1)(law_p).cpu().numpy(), nn.Softmax(dim=-1)(term_p).cpu().numpy(),
                                     nn.Softmax(dim=-1)(accu_p).cpu().numpy(),
                                     record[3].cpu().numpy(), record[5].cpu().numpy(), record[4].cpu().numpy())

        soft_targets = torch.from_numpy(soft_targets).cuda()
        law_logits = torch.log_softmax(law, dim=-1)
        soft_loss = -torch.mean(torch.sum(soft_targets[:, :law_cls] * law_logits, dim=-1))

        accu_logits = torch.log_softmax(accu, dim=-1)
        soft_loss += -torch.mean(torch.sum(soft_targets[:, law_cls:law_cls + accu_cls] * accu_logits, dim=-1))

        term_logits = torch.log_softmax(term, dim=-1)
        soft_loss += -torch.mean(
            torch.sum(soft_targets[:, law_cls + accu_cls:law_cls + accu_cls + term_cls] * term_logits, dim=-1))

        total += len(record[0])
        pred_law = np.argmax(law.cpu().data.numpy(), axis=1)
        target_law = record[3].cpu().data.numpy()
        law_acc += sum(pred_law == target_law) / len(target_law)
        law_loss = loss_func(law, record[3])  # 计算两者的误差

        pred_accu = np.argmax(accu.cpu().data.numpy(), axis=1)
        target_accu = record[4].cpu().data.numpy()
        accu_acc += sum(pred_accu == target_accu) / len(target_accu)
        accu_loss = loss_func(accu, record[4])  # 计算两者的误差

        pred_term = np.argmax(term.cpu().data.numpy(), axis=1)
        target_term = record[5].cpu().data.numpy()
        term_acc += sum(pred_term == target_term) / len(target_term)
        term_loss = loss_func(term, record[5])  # 计算两者的误差

        loss = law_loss
        total_law_loss += law_loss.item()
        loss += accu_loss
        total_accu_loss += accu_loss.item()
        loss += term_loss
        total_term_loss += term_loss.item()

        loss += soft_threshold * soft_loss
        total_soft_loss += soft_loss.item()

        total_loss += loss.item()
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        if step % 100 == 0:
            step += 1
            print("step:", step, "law_acc:", sum(pred_law == target_law) / len(target_law),
                  "accu_acc:", sum(pred_accu == target_accu) / len(target_accu),
                  "term_acc:", sum(pred_term == target_term) / len(target_term),
                  "loss:", total_loss / step,
                  "law_loss", total_law_loss / step,
                  "accu_loss", total_accu_loss / step,
                  "term_loss", total_term_loss / step, 'soft_loss:', total_soft_loss / step)
        law = nn.Softmax(dim=-1)(law)
        accu = nn.Softmax(dim=-1)(accu)
        term = nn.Softmax(dim=-1)(term)
        pred_law = law.cpu().data.numpy()
        if law_preds is None:
            law_preds = pred_law
        else:
            law_preds = np.concatenate((law_preds, pred_law))
        target_law = record[3].cpu().data.numpy()
        law_labels = np.concatenate((law_labels, target_law))

        pred_accu = accu.cpu().data.numpy()
        if accu_preds is None:
            accu_preds = pred_accu
        else:
            accu_preds = np.concatenate((accu_preds, pred_accu))
        target_accu = record[4].cpu().data.numpy()
        accu_labels = np.concatenate((accu_labels, target_accu))

        pred_term = term.cpu().data.numpy()
        if term_preds is None:
            term_preds = pred_term
        else:
            term_preds = np.concatenate((term_preds, pred_term))
        target_term = record[5].cpu().data.numpy()
        term_labels = np.concatenate((term_labels, target_term))
    return {"total loss": total_loss / len(dataloader),
            "law_acc": law_acc / len(dataloader),
            "accu_acc": accu_acc / len(dataloader),
            "term_acc": term_acc / len(dataloader),
            "law_loss": total_law_loss / len(dataloader),
            "accu_loss": total_accu_loss / len(dataloader),
            "term_loss": total_term_loss / len(
                dataloader)}, law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels


def eval_iteration(model, dataloader):
    total_loss = 0
    total_law_loss = 0
    total_accu_loss = 0
    total_term_loss = 0
    law_preds = np.array([])
    term_preds = np.array([])
    accu_preds = np.array([])
    law_labels = np.array([])
    term_labels = np.array([])
    accu_labels = np.array([])

    _law_preds = None
    _term_preds = None
    _accu_preds = None
    tmp = 0
    with torch.no_grad():
        model.eval()
        for record in tqdm(dataloader):
            law, accu, term = model(record[0], record[1], record[2])  # 喂给 net 训练数据 x, 输出分析值

            pred_law = np.argmax(law.cpu().data.numpy(), axis=1)
            law_preds = np.concatenate((law_preds, pred_law))
            target_law = record[3].cpu().data.numpy()
            law_labels = np.concatenate((law_labels, target_law))
            law_loss = loss_func(law, record[3])  # 计算两者的误差

            pred_accu = np.argmax(accu.cpu().data.numpy(), axis=1)
            accu_preds = np.concatenate((accu_preds, pred_accu))
            target_accu = record[4].cpu().data.numpy()
            accu_labels = np.concatenate((accu_labels, target_accu))
            accu_loss = loss_func(accu, record[4])  # 计算两者的误差

            pred_term = np.argmax(term.cpu().data.numpy(), axis=1)
            term_preds = np.concatenate((term_preds, pred_term))
            target_term = record[5].cpu().data.numpy()
            term_labels = np.concatenate((term_labels, target_term))
            term_loss = loss_func(term, record[5])  # 计算两者的误差

            loss = law_loss
            total_law_loss += law_loss.item()
            loss += accu_loss
            total_accu_loss += accu_loss.item()
            loss += term_loss
            total_term_loss += term_loss.item()
            total_loss += loss.item()

            law = nn.Softmax(dim=-1)(law)
            accu = nn.Softmax(dim=-1)(accu)
            term = nn.Softmax(dim=-1)(term)
            pred_law = law.cpu().data.numpy()
            if _law_preds is None:
                _law_preds = pred_law
            else:
                _law_preds = np.concatenate((_law_preds, pred_law))

            pred_accu = accu.cpu().data.numpy()
            if _accu_preds is None:
                _accu_preds = pred_accu
            else:
                _accu_preds = np.concatenate((_accu_preds, pred_accu))

            pred_term = term.cpu().data.numpy()
            if _term_preds is None:
                _term_preds = pred_term
            else:
                _term_preds = np.concatenate((_term_preds, pred_term))

    law_acc = metrics.accuracy_score(law_labels, law_preds)
    law_f1 = metrics.f1_score(law_labels, law_preds, average='macro')
    law_p = metrics.precision_score(law_labels, law_preds, average='macro')
    law_r = metrics.recall_score(law_labels, law_preds, average='macro')

    accu_acc = metrics.accuracy_score(accu_labels, accu_preds)
    accu_f1 = metrics.f1_score(accu_labels, accu_preds, average='macro')
    accu_p = metrics.precision_score(accu_labels, accu_preds, average='macro')
    accu_r = metrics.recall_score(accu_labels, accu_preds, average='macro')

    term_acc = metrics.accuracy_score(term_labels, term_preds)
    term_f1 = metrics.f1_score(term_labels, term_preds, average='macro')
    term_p = metrics.precision_score(term_labels, term_preds, average='macro')
    term_r = metrics.recall_score(term_labels, term_preds, average='macro')
    law_report = get_report(law_labels.astype(int), law_preds.astype(int))
    accu_report = get_report(accu_labels.astype(int), accu_preds.astype(int))
    term_report = get_report(term_labels.astype(int), term_preds.astype(int))

    return {"total loss": total_loss / len(dataloader),
            "law_loss": total_law_loss / len(dataloader),
            "accu_loss": total_accu_loss / len(dataloader),
            "term_loss": total_term_loss / len(dataloader),
            "law_acc": law_acc, "law_f1": law_f1, "law_p": law_p, "law_r": law_r,
            "accu_acc": accu_acc, "accu_f1": accu_f1, "accu_p": accu_p, "accu_r": accu_r,
            "term_acc": term_acc, "term_f1": term_f1, "term_p": term_p, "term_r": term_r,
            "law_report": law_report, "accu_report": accu_report,
            "term_report": term_report}, _law_preds, _term_preds, _accu_preds, law_labels, term_labels, accu_labels


def get_inputs(law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels):
    law2accu = pkl.load(open("law2accu.pkl", 'rb'))
    law_mp_accu = law2accu[law_labels.astype(int)]
    law2term = pkl.load(open("law2term.pkl", "rb"))
    law_mp_term = law2term[law_labels.astype(int)]

    law_targets = torch.zeros_like(torch.from_numpy(law_preds))
    law_targets.scatter_(1, torch.unsqueeze(torch.from_numpy(law_labels).to(int), 1), 1.0)

    accu_targets = torch.from_numpy(law_mp_accu * accu_preds)
    accu_targets.scatter_(1, torch.unsqueeze(torch.from_numpy(accu_labels).to(int), 1), 1.0)

    term_targets = torch.from_numpy(law_mp_term * term_preds)
    term_targets.scatter_(1, torch.unsqueeze(torch.from_numpy(term_labels).to(int), 1), 1.0)

    inputs = np.concatenate([law_preds, accu_preds, term_preds], axis=1)
    targets = torch.cat([law_targets, accu_targets, term_targets], dim=1).numpy()
    return inputs, targets


def predict(model, dataloader):
    law_preds = None
    term_preds = None
    accu_preds = None
    law_labels = np.array([])
    term_labels = np.array([])
    accu_labels = np.array([])
    tmp = 0
    with torch.no_grad():
        model.eval()
        for record in tqdm(dataloader):
            law, accu, term = model(record[0], record[1], record[2])  # 喂给 net 训练数据 x, 输出分析值
            law = nn.Softmax(dim=-1)(law)
            accu = nn.Softmax(dim=-1)(accu)
            term = nn.Softmax(dim=-1)(term)
            pred_law = law.cpu().data.numpy()
            if law_preds is None:
                law_preds = pred_law
            else:
                law_preds = np.concatenate((law_preds, pred_law))
            target_law = record[3].cpu().data.numpy()
            law_labels = np.concatenate((law_labels, target_law))

            pred_accu = accu.cpu().data.numpy()
            if accu_preds is None:
                accu_preds = pred_accu
            else:
                accu_preds = np.concatenate((accu_preds, pred_accu))
            target_accu = record[4].cpu().data.numpy()
            accu_labels = np.concatenate((accu_labels, target_accu))

            pred_term = term.cpu().data.numpy()
            if term_preds is None:
                term_preds = pred_term
            else:
                term_preds = np.concatenate((term_preds, pred_term))
            target_term = record[5].cpu().data.numpy()
            term_labels = np.concatenate((term_labels, target_term))

    return law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def update_inputs(model, dataloader):
    law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels = predict(model, dataloader)
    return get_inputs(law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels)


def update_soft(model, inputs, target):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer.zero_grad()

    law_logits, accu_logits, term_logits = model(inputs)

    law_logits = torch.log_softmax(law_logits, dim=-1)
    loss = -torch.mean(torch.sum(target[:, :law_cls] * law_logits, dim=-1))

    accu_logits = torch.log_softmax(accu_logits, dim=-1)
    loss += -torch.mean(torch.sum(target[:, law_cls:law_cls + accu_cls] * accu_logits, dim=-1))

    term_logits = torch.log_softmax(term_logits, dim=-1)
    loss += -torch.mean(torch.sum(target[:, law_cls + accu_cls:law_cls + accu_cls + term_cls] * term_logits, dim=-1))

    loss.backward()
    optimizer.step()
    return loss.item()


def train_gnn_p(model, loader):
    total_loss = 0
    model.train()
    for i, sample in enumerate(loader):
        loss = update_soft(model, sample[0], sample[1])
        total_loss += loss
    return total_loss / len(loader)


def eval_gnn_p(model, loader):
    total_loss = 0
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(loader):
            loss = eval_gnn(model, sample[0], sample[1])
            total_loss += loss
    return total_loss / len(loader)


def eval_gnn(model, inputs, target):
    model.eval()
    law_logits, accu_logits, term_logits = model(inputs)

    law_logits = torch.log_softmax(law_logits, dim=-1)
    loss = -torch.mean(torch.sum(target[:, :law_cls] * law_logits, dim=-1))

    accu_logits = torch.log_softmax(accu_logits, dim=-1)
    loss += -torch.mean(torch.sum(target[:, law_cls:law_cls + accu_cls] * accu_logits, dim=-1))

    term_logits = torch.log_softmax(term_logits, dim=-1)
    loss += -torch.mean(torch.sum(target[:, law_cls + accu_cls:law_cls + accu_cls + term_cls] * term_logits, dim=-1))
    return loss.item()


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(2021)
    # ***************************************************************#
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
    gnn_p = GNN(1, 1, torch.from_numpy(adj).to(torch.float32).cuda())
    gnn_p.cuda()
    gnn_p.load("gnn_p.p")

    batch_size = 4
    model = BERTLJP(input_fact=1, with_distill=1, mask_stategy=4)
    model.cuda()
    model.load("ToksTransformer114.p")
    train_dataset = CrimeFactDataset("train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_fact
    )
    valid_dataset = CrimeFactDataset("valid")
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_fact
    )
    test_dataset = CrimeFactDataset("test")
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_fact
    )

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': 1e-5}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()
    top_acc = None
    model_name = "EM114"
    best_inputs = []
    for iters in range(1, 5):
        for epoch in range(3):
            if epoch % 3 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.9
            content = model_name + "\n[TRAIN]epoch:%d\n" % epoch
            soft_threshold = 0 if iters == 0 else 0.1
            train_met, law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels = train_iteration(model,
                                                                                                                 gnn_p,
                                                                                                                 train_dataloader,
                                                                                                                 soft_threshold=soft_threshold)
            train_inputs, train_targets = get_inputs(law_preds, term_preds, accu_preds, law_labels, term_labels,
                                                     accu_labels)
            for key in train_met:
                print(key + ":", train_met[key])
                content += key + " : " + str(train_met[key]) + "\n"
            met, law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels = eval_iteration(model,
                                                                                                          valid_dataloader)
            valid_inputs, valid_targets = get_inputs(law_preds, term_preds, accu_preds, law_labels, term_labels,
                                                     accu_labels)
            content += "[VALID]\n"
            print("[VALID]epoch:", epoch)
            if top_acc is None or met['law_acc'] > top_acc:
                top_acc = met['law_acc']
                content += "new top\n"
                print("new top")
                model.save("%s.p" % model_name)
            for key in met:
                print(key + ":", met[key])
                content += key + " : " + str(met[key]) + "\n"
            met, law_preds, term_preds, accu_preds, law_labels, term_labels, accu_labels = eval_iteration(model,
                                                                                                          test_dataloader)
            test_inputs, test_targets = get_inputs(law_preds, term_preds, accu_preds, law_labels, term_labels,
                                                   accu_labels)
            content += "[TEST]\n"
            print("[TEST]epoch:", epoch)
            for key in met:
                print(key + ":", met[key])
                content += key + " : " + str(met[key]) + "\n"
            send_email(content)
            if "new top" in content:
                best_inputs = [train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets]

        gnn_p_train_set = GNNpDataset(best_inputs[0], best_inputs[1])
        gnn_p_train_loader = DataLoader(
            gnn_p_train_set, batch_size=128,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_gnn_p
        )
        gnn_p_valid_set = GNNpDataset(best_inputs[2], best_inputs[3])
        gnn_p_valid_loader = DataLoader(
            gnn_p_valid_set, batch_size=128,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_gnn_p
        )
        gnn_p_test_set = GNNpDataset(best_inputs[4], best_inputs[5])
        gnn_p_test_loader = DataLoader(
            gnn_p_test_set, batch_size=128,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_gnn_p
        )
        best_loss = float('inf')
        for epoch in range(1, 30):
            train_loss = train_gnn_p(gnn_p, gnn_p_train_loader)
            valid_loss = eval_gnn_p(gnn_p, gnn_p_valid_loader)
            print(train_loss, valid_loss)
            if valid_loss < best_loss:
                best_loss = valid_loss
                gnn_p.save("gnn_p.p")
                print("new top")
        gnn_p.load("gnn_p.p")
