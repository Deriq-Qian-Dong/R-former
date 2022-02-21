from sklearn import metrics
from modeling import *
from tqdm import tqdm
import pickle as pkl
import torch.nn
import torch


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


def train_iteration(model, dataloader, BATCH_SIZE=128):
    total_loss = 0
    accu_acc = 0
    law_acc = 0
    term_acc = 0
    total_law_loss = 0
    total_accu_loss = 0
    total_term_loss = 0
    model.train()
    total = 0
    for step, record in enumerate(dataloader):
        law, accu, term = model(record[0], record[1], record[2])  # 喂给 net 训练数据 x, 输出分析值
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
                  "term_loss", total_term_loss / step)
    return {"total loss": total_loss / len(dataloader),
            "law_acc": law_acc / len(dataloader),
            "accu_acc": accu_acc / len(dataloader),
            "term_acc": term_acc / len(dataloader),
            "law_loss": total_law_loss / len(dataloader),
            "accu_loss": total_accu_loss / len(dataloader),
            "term_loss": total_term_loss / len(dataloader)}


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
            "law_report": law_report, "accu_report": accu_report, "term_report": term_report}


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(2021)
    # ***************************************************************#
    batch_size = 4
    model = BERTLJP(input_fact=1, with_distill=1, mask_stategy=4)
    model.cuda()
    # model.load("BERT-finetune.p")
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
        shuffle=True,
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
    # 策略3
    model_name = "ToksTransformer114"
    for epoch in range(1, 5):
        if epoch % 3 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        content = model_name + "\n[TRAIN]epoch:%d\n" % epoch
        train_met = train_iteration(model, train_dataloader)
        for key in train_met:
            print(key + ":", train_met[key])
            content += key + " : " + str(train_met[key]) + "\n"
        met = eval_iteration(model, valid_dataloader)
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
        met = eval_iteration(model, test_dataloader)
        content += "[TEST]\n"
        print("[TEST]epoch:", epoch)
        for key in met:
            print(key + ":", met[key])
            content += key + " : " + str(met[key]) + "\n"
        send_email(content)
