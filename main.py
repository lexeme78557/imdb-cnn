import torch
from torch import optim
from helper import *


if __name__=='__main__':
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    THRESHOLD = 5 
    MAX_LEN = 200
    BATCH_SIZE = 32 
    train_data = torchtext.datasets.IMDB(root='.data', split='train')
    train_data = list(train_data)
    train_data = [(x[0], cnn_preprocess(x[1])) for x in train_data]
    train_data, test_data = train_data[0:10000] + train_data[12500:12500+10000], train_data[10000:12500] + train_data[12500+10000:], 

    # Make pos/neg
    train_data = [('neg' if x[0] == 1 else 'pos', x[1]) for x in train_data]  
    test_data = [('neg' if x[0] == 1 else 'pos', x[1]) for x in test_data]
    train_dataset = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_dataset = TextDataset(test_data, 'test', THRESHOLD, MAX_LEN, train_dataset.idx2word, train_dataset.word2idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    
    cnn_model = CNN(vocab_size = train_dataset.vocab_size,
                embed_size = 128, 
                out_channels = 64, 
                filter_heights = [2, 3, 4], 
                stride = 1, 
                dropout = 0.5, 
                num_classes = 2, 
                pad_idx = train_dataset.word2idx[CNN_PAD])

    # Put your model on the device (cuda or cpu)
    cnn_model = cnn_model.to(DEVICE)
    
    LEARNING_RATE = 5e-4 

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # Define the optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    
    N_EPOCHS = 100
    train_cnn_model(cnn_model, N_EPOCHS, train_loader, optimizer, criterion)
    
    evaluate(cnn_model, test_loader, criterion, use_tqdm=True)