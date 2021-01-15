# coding:utf-8
from __future__ import division
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.config import Config
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo3 import YoloBody
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
from dataloader import YoloDataset, yolo_dataset_collate
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, cuda):
    total_loss = 0.
    val_loss = 0.
    start_time = time.time()
    with tqdm(total=epoch_size, desc='Epoch {}/{}'.format((epoch + 1), Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Total_Loss': total_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc='Epoch {}/{}'.format((epoch + 1), Epoch), postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]

                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'Val_Loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
    net.train()
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    return val_loss / (epoch_size_val + 1)

if __name__ == "__main__":
    # 参数初始化
    train_path = 'BDD100K_train.txt'
    val_path = 'BDD100K_val.txt'
    model = YoloBody(Config)
    Cuda = True
    Resume = False
    Use_Data_Loader = True

    # 加快模型训练的效率
    print('Loading weights into state dict...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/yolo_weights.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    with open(train_path) as f1:
        lines1 = f1.readlines()
    with open(val_path) as f2:
        lines2 = f2.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines1)
    np.random.seed(None)
    np.random.seed(10101)
    np.random.shuffle(lines2)
    np.random.seed(None)

    num_val = int(len(lines2))
    num_train = int(len(lines1))

    early_stopping = EarlyStopping(patience=7, verbose=True)

    if True:
        # 最开始使用1e-3的学习率可以收敛的更快
        lr = 1e-3
        Batch_size = 32
        Init_Epoch = 0
        Freeze_Epoch = 20

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines1, (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines2, (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=1, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=1, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines1,
                            (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines2,
                            (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        # 冻结一定部分训练
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            valloss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 16
        Freeze_Epoch = 20
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        if Resume:
            print('Resume from checkpoint...')
            checkpoint = torch.load('./checkpoint/checkpoint.pkl')
            Init_Epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.last_epoch = Init_Epoch - Freeze_Epoch

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines1, (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines2, (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=1, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=1, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines1,
                            (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines2,
                            (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        # 解冻后训练
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            valloss = fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            path_checkpoint = './checkpoint/checkpoint.pkl'
            torch.save(checkpoint, path_checkpoint)
            early_stopping(valloss)
            if early_stopping.early_stop:
                print('Early Stopping!')
                break