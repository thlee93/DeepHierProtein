import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class HierMetric(nn.Module):    
    def __init__(self, charset_size, latent_dim=15, num_hidden=15, num_classes=88):
        super(HierMetric, self).__init__()
        
        self.num_windows = [256, 256, 256, 256, 256, 256, 256, 256]
        self.len_windows = [8, 12, 16, 20, 24, 28, 32, 36]

        # number of features from convolutional motif detector
        n_convfeatures = np.sum(self.num_windows)
        # list of features extracted from each variable-length filters
        self.motifs = nn.ModuleList()

        for len_w, num_w in zip(self.len_windows, self.num_windows):
            # each motif feature is extracted through conv -> one-max pooling
            motif_cnn = nn.Sequential( nn.Conv1d(charset_size, num_w, len_w), 
                                                 nn.BatchNorm1d(num_w),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(1000-len_w+1) ) 
            torch.nn.init.xavier_normal_(motif_cnn[0].weight)
            self.motifs.append(motif_cnn)

        self.embedding = nn.Linear( n_convfeatures, latent_dim )

        self.cls_classifier = nn.Sequential( nn.Linear(latent_dim, num_hidden),
                                             nn.Linear(num_hidden, 5) )

        self.fam_classifier = nn.Sequential( nn.Linear(latent_dim, num_hidden),
                                             nn.Linear(num_hidden, 40) )

        self.sub_classifier = nn.Sequential( nn.Linear(latent_dim, num_hidden),
                                             nn.Linear(num_hidden, 86) )


    def forward(self, x):
        features = []

        # extract features from each convolutional motif detector
        for m_feature in self.motifs :
            features.append( m_feature(x) )

        out = torch.cat( features, 1 )
        out = out.view(-1, out.size(1))

        hidden = self.embedding(out)
        cls = self.cls_classifier(hidden)
        fam = self.fam_classifier(hidden)
        sub = self.sub_classifier(hidden)

        return cls, fam, sub, hidden

if __name__ == '__main__':
    pass
