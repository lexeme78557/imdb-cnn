import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, filter_heights, stride, dropout, num_classes, pad_idx):
        super(CNN, self).__init__()
        
        self.emb_layer = nn.Embedding(vocab_size, embed_size, pad_idx)
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, out_channels, (fh, embed_size), stride) for fh in filter_heights])
        self.drop_layer = nn.Dropout(dropout)
        self.lin_layer = nn.Linear(out_channels * len(filter_heights), num_classes)

    def forward(self, texts):
        """
        texts: LongTensor [batch_size, max_len]
        
        Returns output: Tensor [batch_size, num_classes]
        """

        out1 = self.emb_layer(texts)
        cat_out = torch.Tensor()
        out1 = torch.unsqueeze(out1, 1)

        for (layer) in (self.conv_layers):
            out = layer(out1)
            out = torch.squeeze(out, 3)
            out = F.relu(out)
            out = torch.max(out, 2).values
            cat_out = torch.cat((cat_out, out), 1)

        out = self.drop_layer(cat_out)
        out = self.lin_layer(out)
        return out