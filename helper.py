import torchtext
import random
import torch
from tqdm.notebook import tqdm
from dataset import *
from model import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, labels):
    """
    Returns accuracy per batch
    output: Tensor [batch_size, n_classes]
    labels: LongTensor [batch_size]
    """
    preds = output.argmax(dim=1) # find predicted class
    correct = (preds == labels).sum().float() # convert into float for division 
    acc = correct / len(labels)
    return acc

import random

def evaluate(model, data_loader, criterion, use_tqdm=False):
    print('Evaluating performance on the test dataset...')
    has_printed=False
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    iterator = tqdm(data_loader) if use_tqdm else data_loader
    total = 0
    for texts, labels in iterator:
        bs = texts.shape[0]
        total += bs
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)
        
        output = model(texts)
        acc = accuracy(output, labels) * len(labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels) * len(labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if random.random() < 0.0015 and bs == 1:
            if not has_printed: print("\nSOME PREDICTIONS FROM THE MODEL:")
            print("Input: "+' '.join([data_loader.dataset.idx2word[idx] for idx in texts[0].tolist() if idx not in {data_loader.dataset.word2idx[CNN_PAD], data_loader.dataset.word2idx[CNN_END]}]))
            print("Prediction:", pred.item(), '\tCorrect Output:', labels.item(), '\n')
            has_printed=True

    full_acc = 100*epoch_acc/total
    full_loss = epoch_loss/total
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return predictions, full_acc, full_loss


def train_cnn_model(model, num_epochs, data_loader, optimizer, criterion):
    print('Training Model...')
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        for texts, labels in tqdm(data_loader):
            texts = texts.to(DEVICE) # shape: [batch_size, MAX_LEN]
            labels = labels.to(DEVICE) # shape: [batch_size]

            optimizer.zero_grad()

            output = model(texts)
            acc = accuracy(output, labels)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, epoch_loss/len(data_loader), 100*epoch_acc/len(data_loader)))
    print('Model Trained!\n')


def cnn_preprocess(review):
    '''
    Simple preprocessing function.
    '''
    res = []
    for x in review.split(' '):
        remove_beg=True if x[0] in {'(', '"', "'"} else False
        remove_end=True if x[-1] in {'.', ',', ';', ':', '?', '!', '"', "'", ')'} else False
        if remove_beg and remove_end: res += [x[0], x[1:-1], x[-1]]
        elif remove_beg: res += [x[0], x[1:]]
        elif remove_end: res += [x[:-1], x[-1]]
        else: res += [x]
    return res
