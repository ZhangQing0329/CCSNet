import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from net.CCSNet import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import torch.nn as nn

tmp_path ='./see'

file = open("log/CCSNet.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True


bce_loss = nn.BCEWithLogitsLoss(weight = None, reduction='mean')

def structure_loss(pred, mask):
    bce   = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou   = 1-(inter+1)/(union-inter+1)


    return bce+iou.mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()
    
def bce2d(input, target):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg
    bce_loss_edge = nn.BCEWithLogitsLoss(weight = weights, size_average=True)
    #bce_loss_edge = nn.BCELoss(weight = weights, size_average=True)
    return bce_loss_edge(input, target)


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record4, loss_record3, loss_record2, loss_record1, loss_record5, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_5, edge_map = model(images)
        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss1 = structure_loss(lateral_map_1, gts)
        loss5 = structure_loss(lateral_map_5, gts)
        losse = dice_loss(edge_map, edges)

        loss = loss4 + loss3 + loss2 + loss1 + losse + loss5 
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record4.update(loss4.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_record5.update(loss5.data, opt.batchsize)
        loss_recorde.update(losse.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-4: {:.4f}], [lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record4.avg, loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
                        

    save_path = 'checkpoints-CCSNet/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if epoch >= 29:
        torch.save(model.state_dict(), save_path + 'CCSNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'CCSNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'CCSNet-%d.pth' % epoch + '\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=60, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=448, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='/home/fiona/Desktop/datasets/COD/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='CCSNet')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()
