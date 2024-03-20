import torch
import torch.nn as nn
import torch.nn.functional as F


def poly1_cross_entropy_torch(logits, labels, class_number=3, epsilon=1.0):
    poly1 = torch.sum(F.one_hot(labels, class_number).float() * F.softmax(logits), dim=-1)
    ce_loss = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction='none')
    poly1_ce_loss = ce_loss + epsilon * (1 - poly1)
    return poly1_ce_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5):
        super(FocalLoss, self).__init__()
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += alpha
        self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        self.alpha = self.alpha.to(logits.device)
        logits_logsoft = F.log_softmax(logits, dim=1)
        logits_softmax = torch.exp(logits_logsoft)
        logits_softmax = logits_softmax.gather(1, labels.view(-1, 1))
        logits_logsoft = logits_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - logits_softmax), self.gamma), logits_logsoft)
        loss = torch.mul(self.alpha, loss.t())[0, :]
        return loss


def poly1_focal_loss_torch(logits = None, labels = None, alpha=0.25, gamma=2, num_classes=5, epsilon=1.0):
    focal_loss_func = FocalLoss(alpha, gamma, num_classes)
    focal_loss = focal_loss_func(logits, labels)

    p = torch.nn.functional.sigmoid(logits)
    labels = torch.nn.functional.one_hot(labels, num_classes)
    labels = torch.tensor(labels, dtype=torch.float32)
    poly1 = labels * p + (1 - labels) * (1 - p)
    poly1_focal_loss = focal_loss + torch.mean(epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
    return poly1_focal_loss


if __name__ == '__main__':
    logits = [[2, 0.5, 1],
              [0.1, 1, 3]]
    labels = [1, 2]
    print("PyTorch loss result:")
    ce_loss = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction='none')
    print("torch cross_entropy:", ce_loss)

    poly1_ce_loss = poly1_cross_entropy_torch(torch.tensor(logits), torch.tensor(labels), class_number=3, epsilon=1.0)
    print("torch poly1_cross_entropy:", poly1_ce_loss)

    focal_loss_func = FocalLoss(alpha=0.25, gamma=2, num_classes=3)
    fc_loss = focal_loss_func(torch.tensor(logits), torch.tensor(labels))
    print("torch focal_loss:", fc_loss)

    poly1_fc_loss = poly1_focal_loss_torch(torch.tensor(logits), torch.tensor(labels), alpha=0.25, gamma=2,
                                           num_classes=3, epsilon=1.0)
    print("torch poly1_focal_loss:", poly1_fc_loss)