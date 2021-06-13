import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from model import TranE
from prepare_data import TrainSet, TestSet
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()

# Required parameters
# NOTE: train tasks and val tasks cannot take command line arguments
parser.add_argument('--lr', default=0.02, type=float)
args = parser.parse_args()
device = torch.device('cuda')
embed_dim = 50
num_epochs = 50
train_bacatch_size = 32
test_batch_size = 256
lr = args.lr
momentum = 0
gamma = 1
d_norm = 2
top_k = 10


def main():
    entity2label = pd.read_csv('./Wiki15k/entity2label.txt', sep='\t', header=None,
                           names=['entity', 'name'],
                           keep_default_na=False, encoding='utf-8')
    entity2label = entity2label.set_index('entity').squeeze().T.to_dict()
    relation2label = pd.read_csv('./Wiki15k/relation2label.txt', sep='\t', header=None,
                               names=['relation', 'name'],
                               keep_default_na=False, encoding='utf-8')
    relation2label = relation2label.set_index('relation').squeeze().T.to_dict()
    train_dataset = TrainSet()
    test_dataset = TestSet()
    test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                       test_dataset.raw_data)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    transe = TranE(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim, d_norm=d_norm,
                   gamma=gamma).to(device)
    optimizer = optim.SGD(transe.parameters(), lr=lr, momentum=momentum)
    for epoch in range(num_epochs):
        # e <= e / ||e||
        entity_norm = torch.norm(transe.entity_embedding.weight.data, dim=1, keepdim=True)
        transe.entity_embedding.weight.data = transe.entity_embedding.weight.data / entity_norm
        total_loss = 0
        for batch_idx, (pos, neg) in enumerate(train_loader):
            pos, neg = pos.to(device), neg.to(device)
            # pos: [batch_size, 3] => [3, batch_size]
            pos = torch.transpose(pos, 0, 1)
            # pos_head, pos_relation, pos_tail: [batch_size]
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
            neg = torch.transpose(neg, 0, 1)
            # neg_head, neg_relation, neg_tail: [batch_size]
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            loss = transe(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}, loss = {total_loss/train_dataset.__len__()}")
        corrct_test = 0
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            # data: [batch_size, 3] => [3, batch_size]
            data = torch.transpose(data, 0, 1)
            corrct_test += transe.tail_predict(data[0], data[1], data[2], k=top_k)
        print(f"===>epoch {epoch+1}, test accuracy {corrct_test/test_dataset.__len__()}")

    corrct_test_top1_head = 0
    corrct_test_top1_tail = 0
    corrct_test_top3_head = 0
    corrct_test_top3_tail = 0
    corrct_test_top10_head = 0
    corrct_test_top10_tail = 0
    mrr_average = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        # data: [batch_size, 3] => [3, batch_size]
        data = torch.transpose(data, 0, 1)

        corrct_test_top1_tail += transe.tail_predict(data[0], data[1], data[2], k=1)
        corrct_test_top1_head += transe.head_predict(data[0], data[1], data[2], k=1)
        corrct_test_top3_tail += transe.tail_predict(data[0], data[1], data[2], k=3)
        corrct_test_top3_head += transe.head_predict(data[0], data[1], data[2], k=3)
        corrct_test_top10_tail += transe.tail_predict(data[0], data[1], data[2], k=10)
        corrct_test_top10_head += transe.head_predict(data[0], data[1], data[2], k=10)
        mrr_average.append(transe.mrr(data[0], data[1], data[2]))
    print(f"Top 1 test accuracy {(corrct_test_top1_tail + corrct_test_top1_head) /test_dataset.__len__()}")
    print(f"Top 3 test accuracy {(corrct_test_top3_tail + corrct_test_top3_head)  / test_dataset.__len__()}")
    print(f"Top 10 test accuracy {(corrct_test_top10_tail + corrct_test_top10_head)  / test_dataset.__len__()}")
    print(f"MRR {np.mean(mrr_average)}")

    def print_relation(head, relatioon, tail, top_10_index):
        head = train_dataset.index_to_entity[head]
        tail = train_dataset.index_to_entity[tail]
        relatioon = train_dataset.index_to_relation[relatioon]
        top_10_index_list = []
        for i in range(10):
            top_10_index_list.append(train_dataset.index_to_entity[top_10_index[i]])

        print(f'------- {entity2label[head]} + {relation2label[relatioon]} = {entity2label[tail]} -----------')
        for i in range(10):
            ind = top_10_index_list[i]
            print(f'{ind} : {entity2label[ind]}')
        print('*'*10)

    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        # data: [batch_size, 3] => [3, batch_size]
        data_ = torch.transpose(data, 0, 1)
        top10_index = transe.tail_top_10(data_[0], data_[1], data_[2])

    for i in range(3):
        head, relatioon, tail = data[i].cpu().numpy()
        top_10_index = top10_index[i].cpu().numpy()
        print_relation(head, relatioon, tail, top_10_index)

    # find relation ship
    print('--'*20)
    print('find nearest entitity')
    top10_index = transe.find_nearest_entities(data_[0])

    def print_e_relation(head,  top_10_index):
        head = train_dataset.index_to_entity[head]
        top_10_index_list = []
        for i in range(10):
            top_10_index_list.append(train_dataset.index_to_entity[top_10_index[i]])

        print(f'-------- find : {entity2label[head]} ---------')
        for i in range(10):
            ind = top_10_index_list[i]
            print(f'{ind} : {entity2label[top_10_index_list[i]]}')
        print('*'*10)

    for i in range(3):
        head, _, _ = data[i].cpu().numpy()
        top_10_index = top10_index[i].cpu().numpy()
        print_e_relation(head, top_10_index)



if __name__ == '__main__':
    main()
