import torch
import torch.nn as nn
import torch.nn.functional as F

# loss['bce'] = balanced_binary_cross_entropy(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])
def balanced_binary_cross_entropy(logits, labels, mask, weights):
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

class compute_losses(nn.Module):
    def __init__(self):
        super(compute_losses, self).__init__()
        self.compute_transform_losses = nn.L1Loss()
        self.class_weights_1 = torch.ones(14,)
        self.class_weights_2 = torch.ones(14,)
    # opt.type can only be chosen from ["both","static","dynamic"]
    def forward(self, outputs,  gt_semantic_seg, features, retransform_features):
        losses = {}
        losses_topview_loss = 0.
        losses_transform_topview_loss = 0.
        losses_transform_loss = 0.
        # logits:outputs,labels:gt_semantic_seg
        gt_semantic_seg = gt_semantic_seg.squeeze(1).bool()
        
        self.class_weights_1 = self.class_weights_1.to(outputs["topview"])
        self.class_weights_2 = self.class_weights_2.to(outputs["transform_topview"])
        losses_topview_loss = balanced_binary_cross_entropy(outputs["topview"],gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights_1)
        losses_transform_topview_loss = balanced_binary_cross_entropy(
            outputs["transform_topview"],gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights_2)
        losses_transform_loss = self.compute_transform_losses(
            features,retransform_features)

        # losses_topview_loss = loss_BCE,losses_transform_loss = loss_cycle,losses_transform_topview_loss = loss_discriminator
        losses["loss"] = losses_topview_loss+ 0.001 * losses_transform_loss + 1* losses_transform_topview_loss

        # acc_seg, only to evaluate, so we use detach here
        losses['acc_seg'] = iou(outputs["topview"].detach().sigmoid()>0.5, gt_semantic_seg[:,:-1,...], gt_semantic_seg[:,-1,...])
        # return losses['loss']
        return losses

def iou(preds, labels, mask=None, per_class=False):
    num_class = preds.shape[1]
    preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)
    labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)
    if mask is not None:
        preds = preds[:, mask.flatten()]
        labels = labels[:, mask.flatten()]
    true_pos = preds & labels
    false_pos = preds & ~labels
    false_neg = ~preds & labels
    tp = true_pos.long()
    fp = false_pos.long()
    fn = false_neg.long()
    if not per_class:
        return tp.sum().float() / (tp.sum() + fn.sum() + fp.sum()).float()
    else:
        valid = labels.int().sum(-1)>0
        return tp.sum(-1), fp.sum(-1), fn.sum(-1), valid

class simple_loss():
    def __init__(self,priors=[0.002,0.008]):
        super(simple_loss, self).__init__()
        self.compute_transform_losses = nn.L1Loss()
        self.priors = torch.tensor(priors)
        self.class_weights = torch.sqrt(1 / self.priors)
    def forward(self, outputs,  gt_semantic_seg, features, retransform_features):
        losses = {}
        losses_topview_loss = 0.
        losses_transform_loss = 0.
        # logits:outputs,labels:gt_semantic_seg
        gt_semantic_seg = gt_semantic_seg.squeeze(1).bool()
        self.class_weights = self.class_weights.to(outputs)
        losses_topview_loss = balanced_binary_cross_entropy(outputs,gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights)
        losses_transform_loss = self.compute_transform_losses(features,retransform_features)
        losses["loss"] = losses_topview_loss + 0.001 * losses_transform_loss
        # acc_seg, only to evaluate, so we use detach here
        losses['acc_seg'] = iou(outputs.detach().sigmoid()>0.5, gt_semantic_seg[:,:-1,...], gt_semantic_seg[:,-1,...])
        return losses