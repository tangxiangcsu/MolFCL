import torch.nn as nn
import torch
import logging
logger = logging.getLogger()

class NCESoftmaxLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, similarity):
        batch_size = similarity.size(0) // 2
        label = torch.tensor([(batch_size + i) % (batch_size*2) for i in range(batch_size*2)]).cuda().long()
        loss = self.criterion(similarity, label)
        return loss


class FlatNCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, similarity):
        pass

