import torch
import torch.nn as nn
import torch.nn.functional as F

def prior_uncertainty_loss(x, mask, priors):
    # priors shape: [2]-->[1,2,1,1]-->[bs,2,196,200]
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    # F.binary_cross_entropy_with_logits(x, priors, reduce=False) return a tensor with the shape of x, i.e. [bs,2,196,200]
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean()

def balanced_binary_cross_entropy(logits, labels, mask, weights):
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

def occupancyloss(logits, labels, mask):
    priors=[0.04]
    xent_weight=1.0
    uncert_weight=0.001

    priors = torch.tensor(priors)
    class_weights = torch.sqrt(1 / priors)

    # Compute binary cross entropy loss
    # only to load self.class_weights and logits to the same device
    class_weights = class_weights.to(logits)
    logits = logits.cpu()
    labels = labels.cpu()
    mask = mask.cpu()
    class_weights = class_weights.cpu()
    bce_loss = balanced_binary_cross_entropy(logits, labels, mask, class_weights)

    # Compute uncertainty loss for unknown image regions
    priors = priors.to(logits)
    uncert_loss = prior_uncertainty_loss(logits, mask, priors)
    return bce_loss * xent_weight + uncert_loss * uncert_weight