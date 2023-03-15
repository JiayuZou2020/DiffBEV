import torch
def iou(preds, labels, mask=None, per_class=False):
    num_class = preds.shape[1]
    # preds.shape:[n,c,h,w]   labels.shape:[n,c,h,w]  mask.shape:[n,1,h,w]
    preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)   # preds.shape:[c,n x h x w]
    labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1) # labelss.shape:[c,n x h x w]
    # mask is used to discriminate background and foreground 
    if mask is not None:
        # ignore background both in preds and labels
        preds = preds[:, mask.flatten()]
        labels = labels[:, mask.flatten()]
    true_pos = preds & labels
    false_pos = preds & ~labels
    false_neg = ~preds & labels
    tp = true_pos.long()
    fp = false_pos.long()
    fn = false_neg.long()
    # tp.sum() means all the tp in all the images,tp.sum(-1) means tp in each class
    if not per_class:
        return tp.sum().float() / (tp.sum() + fn.sum() + fp.sum()).float()
    else:
        # valids: only to choose the positive samples,tp.sum(-1).shape: [1,num_classes]
        valid = labels.int().sum(-1)>0
        return tp.sum(-1), fp.sum(-1), fn.sum(-1), valid
