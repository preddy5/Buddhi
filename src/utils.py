import torch

THRESHOLD = 1
def to_one_hot(label, n_classes=10):
    bs = label.shape[0]
    one_hot_labels = torch.zeros([bs, n_classes], )# + 0.00001
    one_hot_labels[torch.arange(bs), label] = 1
    return one_hot_labels

def update_progress(bar, info):
    message = ''
    for k in info.keys():
        message += '{}: {:.4f} '.format(k, info[k])
    bar.set_description(message)
    bar.refresh() # to show immediately the update

def detach_state(state):
    state_ = []
    for s in state:
        if type(s)!=int:
            state_.append(s.detach())
        else:
            state_.append(s)
    return state_

def geo_sum(n, alpha, subtract1=False):
    if subtract1:
        n = n-1
    num = 1 - alpha**n
    denom = 1 - alpha
    return num/denom

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)#.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
