import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from .loss import attention_consistency_loss, final_embedding_entropy_loss
from collections import defaultdict

# def train_or_val(loader, device, epoch, args, optimizer, model, criterion_CN, scaler, scheduler, isTrain):

#     losses = defaultdict(float)
#     step, epoch_loss, metric_count, num_correct = (0,0,0,0)

#     for batched in loader:

#         step += 1
#         optimizer.zero_grad()
#         loss = 0
        
#         with torch.cuda.amp.autocast():
                    
#             data, labels = batched[0].to(device), batched[1].to(device)
#             if args.model == 'RGNN':
#                 output, final_embedding = model(data)
#             else:
#                 output = model(data)
            
#             # Compute reconstruction loss
#             if args.lambda_XENT:
#                 XENT_loss = args.lambda_XENT * criterion_CN(output, labels)
#                 loss += XENT_loss
#                 losses['XENT'] += XENT_loss.item()

#             if args.lambda_AC:
#                 AC_loss = args.lambda_AC * attention_consistency_loss(final_embedding, model.prediction.weight, labels)
#                 loss += AC_loss
#                 losses['AC'] += AC_loss.item()
            
#             if args.lambda_ES:
#                 es_loss = args.lambda_ES * final_embedding_entropy_loss(final_embedding, model.prediction.weight, labels)
#                 loss += es_loss
#                 losses['ES'] += es_loss.item()

#         if isTrain:
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         epoch_loss += loss
    
#         # Count num_correct
#         value = torch.eq(output.argmax(dim=1), labels)
#         metric_count += len(value)        
#         num_correct += value.sum().item()
        
#     # Summarize metrics and return
#     metric = num_correct / metric_count
#     epoch_loss /= step
    
#     for key in losses.keys():
#         losses[key] /= step
        
#     losses['acc'] = metric
#     scheduler.step(epoch)

#     return losses

def train_or_val(loader, device, epoch, args, optimizer, model, criterion_CN, scaler, scheduler, isTrain):
    losses = defaultdict(float)
    step, epoch_loss, metric_count, num_correct = (0, 0, 0, 0)

    for batched in loader:
        step += 1
        loss = 0

        if isTrain:
            optimizer.zero_grad()
            # Training: gradients are needed.
            with torch.cuda.amp.autocast():
                data, labels = batched[0].to(device), batched[1].to(device)
                if args.model == 'RGNN' or args.model == 'ViT':
                    output, final_embedding = model(data)
                else:
                    output = model(data)
                
                # Compute losses
                if args.lambda_XENT:
                    XENT_loss = args.lambda_XENT * criterion_CN(output, labels)
                    loss += XENT_loss
                    losses['XENT'] += XENT_loss.item()

                if args.lambda_AC:
                    AC_loss = args.lambda_AC * attention_consistency_loss(final_embedding, model.prediction.weight, labels)
                    loss += AC_loss
                    losses['AC'] += AC_loss.item()

                if args.lambda_ES:
                    es_loss = args.lambda_ES * final_embedding_entropy_loss(final_embedding, model.prediction.weight, labels)
                    loss += es_loss
                    losses['ES'] += es_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Validation: no gradient computation needed.
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    data, labels = batched[0].to(device), batched[1].to(device)
                    if args.model == 'RGNN' or args.model == 'ViT':
                        output, final_embedding = model(data)
                    else:
                        output = model(data)
                    
                    # Compute losses similarly (but no backward pass)
                    if args.lambda_XENT:
                        XENT_loss = args.lambda_XENT * criterion_CN(output, labels)
                        loss += XENT_loss
                        losses['XENT'] += XENT_loss.item()

                    if args.lambda_AC:
                        AC_loss = args.lambda_AC * attention_consistency_loss(final_embedding, model.prediction.weight, labels)
                        loss += AC_loss
                        losses['AC'] += AC_loss.item()

                    if args.lambda_ES:
                        es_loss = args.lambda_ES * final_embedding_entropy_loss(final_embedding, model.prediction.weight, labels)
                        loss += es_loss
                        losses['ES'] += es_loss.item()

        epoch_loss += loss

        # Count correct predictions
        value = torch.eq(output.argmax(dim=1), labels)
        metric_count += len(value)        
        num_correct += value.sum().item()

    # Summarize metrics
    metric = num_correct / metric_count
    epoch_loss /= step
    for key in losses.keys():
        losses[key] /= step
    losses['acc'] = metric

    scheduler.step(epoch)

    return losses
