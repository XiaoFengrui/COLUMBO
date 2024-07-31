import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from model import *

from tools import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time

# 训练函数
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    correct = 0
    overall = 0
    model.train()
    
    for batch in tqdm(iterator):
        text, label = batch
        logits = model(text)
        probs = F.softmax(logits, dim=1).squeeze(dim=0)
        # print(probs.shape, torch.argmax(probs, dim=1).shape)
        pred = torch.argmax(probs, dim=1)
        # print(logits, label)
        loss = criterion(logits, label)
        # acc = binary_accuracy(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        correct += torch.sum(pred == label).item()
        overall += len(text)
        
    return epoch_loss / len(iterator), correct / overall

# 评估函数
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    correct = 0
    overall = 0
    model.eval()

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, label = batch
            logits = model(text)
            probs = F.softmax(logits, dim=1).squeeze(dim=0)
            # print(probs.shape, torch.argmax(probs, dim=1).shape)
            pred = torch.argmax(probs, dim=1)
            loss = criterion(logits, label)
            # acc = binary_accuracy(pred, label)
            
            epoch_loss += loss.item()
            # 存储当前批次的预测和标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    # 计算并输出总的指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print('Acc: %.6lf\tPre: %.6lf\tRec: %.6lf\tF1: %.6lf'%(acc, prec, rec, f1))
        
    return epoch_loss / len(iterator), acc

unk_cnt = 0
# 处理文本和标签函数
def process_text(text, word2idx, device, max_len=128):
    global unk_cnt
    tokens = split_word_tokenize(text)
    if len(tokens) > max_len:
        padded_tokens = tokens[:max_len]
    else:
        padded_tokens = tokens + ['<pad>'] * (max_len - len(tokens))
    input_id = [word2idx.get(token, word2idx['<unk>']) for token in padded_tokens]
    unk_cnt += input_id.count(1)
    input_id = torch.tensor(input_id).to(device)

    return input_id

def process_labels(label, device):
    return torch.tensor(label, dtype=torch.long).to(device)

def main():
    # load payload
    dataset = 'SIK'
    training_ratio = 20
    # dataset = 'HPD'
    print('... loading data from %s...'%dataset)
    train_df = pd.read_csv('/home/ustc-5/XiaoF/AdvWebDefen/Dataset/' + dataset + '/fewshot/tr_'+ str(training_ratio) + '/train.tsv', delimiter='\t', names=['text', 'label'], skiprows=1)
    # train_df, _ = train_test_split(train_df, train_size=float(training_ratio/80), stratify=train_df['label'])
    val_df = pd.read_csv('/home/ustc-5/XiaoF/AdvWebDefen/Dataset/' + dataset + '/dev.tsv', delimiter='\t', names=['text', 'label'], skiprows=1)
    # val_df = val_df.iloc[:100]
    test_df = pd.read_csv('/home/ustc-5/XiaoF/AdvWebDefen/Dataset/' + dataset + '/test.tsv', delimiter='\t', names=['text', 'label'], skiprows=1)
    # test_df = test_df.iloc[:100]
    # print(train_df.iloc[:10], val_df.iloc[:10])
    # exit(0)

    word2idx = load_word2idx('/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/lstm/word2idx.json')
    # print(word2idx['<unk>'], unk_cnt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 处理文本和标签
    print('... load train ...')
    train_texts = [process_text(text, word2idx, device) for text in tqdm(train_df['text'])]
    train_labels = [process_labels(label, device) for label in tqdm(train_df['label'])]
    # print(len(train_texts), len(train_labels))
    train_dataset = TensorDataset(torch.stack(train_texts), torch.stack(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print('... load validate ...')
    val_texts = [process_text(text, word2idx, device) for text in tqdm(val_df['text'])]
    val_labels = [process_labels(label, device) for label in tqdm(val_df['label'])]
    val_dataset = TensorDataset(torch.stack(val_texts), torch.stack(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    print('... load test ...')
    test_texts = [process_text(text, word2idx, device) for text in tqdm(test_df['text'])]
    test_labels = [process_labels(label, device) for label in tqdm(test_df['label'])]
    test_dataset = TensorDataset(torch.stack(test_texts), torch.stack(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    print('<UNK> count: ', unk_cnt)
    # exit(0)

    # 初始化模型
    model, optimizer = initilize_lstm_model(device, vocab_size=len(word2idx), embed_dim=300, dropout=0.5, learning_rate=0.01)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # 训练模型
    N_EPOCHS = 20
    best_acc = 0

    for epoch in range(N_EPOCHS):
        print(f'Training Epoch: {epoch+1:02}')

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        if valid_acc > best_acc:
            # torch.save(model.state_dict(), './lstm_model_' + dataset + '.pth')
            torch.save(model.state_dict(), './lstm_model_' + dataset + '_%d.pth'%(training_ratio))
    
    print("... Testing ...")
    model, _ = initilize_lstm_model(device, vocab_size=len(word2idx), embed_dim=300, dropout=0.5)
    # model.load_state_dict(torch.load('./lstm_model_' + dataset + '.pth'))
    model.load_state_dict(torch.load('./lstm_model_' + dataset + '_%d.pth'%(training_ratio)))
    _, _ = evaluate(model, test_loader, criterion)

if __name__=='__main__':
    # main()
    
    dataset = 'SIK' # th = 2.75
    test_dataset = 'HPD'
    # dataset = 'HPD' # th = 5
    # train(dataset)
    t1 = time.time()
    word2idx = load_word2idx('/home/ustc-5/XiaoF/AdvWebDefen/AutoSpear-main/classifiers/repos/lstm/word2idx.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = initilize_lstm_model(device, vocab_size=len(word2idx), embed_dim=300, dropout=0.5)
    model.load_state_dict(torch.load('./lstm_model_' + dataset + '.pth'))
    
    test_df = pd.read_csv('/home/ustc-5/XiaoF/AdvWebDefen/Dataset/' + test_dataset + '/test.tsv', delimiter='\t', names=['text', 'label'], skiprows=1)
    test_texts = [process_text(text, word2idx, device) for text in tqdm(test_df['text'])]
    test_labels = [process_labels(label, device) for label in tqdm(test_df['label'])]
    test_dataset = TensorDataset(torch.stack(test_texts), torch.stack(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    _, _ = evaluate(model, test_loader, criterion)
    print('time cost: ', time.time() - t1)