import os
import sys
from collections import OrderedDict
from copy import deepcopy
import torch

import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import CharbonnierLoss
from model import *
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

torch.backends.cudnn.benchmark = True
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import argparse

class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=2, help='batch size')
        parser.add_argument('--nepoch', type=int, default=1000, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=0, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=0, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='light')
        parser.add_argument('--base_model', type=str,
                            default=r'D:\zjh\GUIDE\GUIDE-code\marine.pth',
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='restormer', help='archtechture')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
        parser.add_argument('--resume', action='store_true', default=False)

        parser.add_argument('--modify', action='store_true', default=True, help='if qkv modify')
        parser.add_argument('--adapter', type=str,
                            default=r'',
                            help='path of pretrained adapter weights')
        parser.add_argument('--modify_resume', action = 'store_true',default =True, help = 'modify resume')




        # args for saving
        parser.add_argument('--base_task', type=str, default='turbid', help='Base task directory')
        parser.add_argument('--adapter_task', type=str, default='marine', help='Adapter task directory')
        parser.add_argument('--save_dir', type=str, default=r'', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')

        parser.add_argument('--train_dir', type=str, default=r'',
                            help='dir of train data')
        parser.add_argument('--val_dir', type=str, default=r'', help='dir of val data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='epochs for warmup')

        # ddp
        parser.add_argument("--local_rank", type=int, default=-1,
                            help='DDP parameter, do not modify')  # 不需要赋值，启动命令 torch.distributed.launch会自动赋值
        parser.add_argument("--distribute", action='store_true', help='whether using multi gpu train')
        parser.add_argument("--distribute_mode", type=str, default='DDP', help="using which mode to ")
        return parser


######### parser ###########
opt = Options().init(argparse.ArgumentParser(description='Image color')).parse_args()
# print(opt)

import utils
from dataset.dataset import *

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu




def train_LN(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.LayerNorm):
            model._modules[name].train()
            for param in model._modules[name].parameters():
                param.requires_grad = True
        train_LN(layer)
    return model


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights,map_location='cpu')
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,strict=True)


def printmodel(model,path):
    with open(path,'w') as f:
        f.write('uformer setting\n')

        for name, param in model.named_parameters():
            f.write(str(name) +' ' + str(param.requires_grad) + ' ' + str(param.numel())+ '\n\n')
        # for name, layer in model.named_children():
        #     f.write(str(name) +' ' + str(layer) + '\n\n')



def add_ghostnet(model, model_type="mixdepth"):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if model_type == "mixdepth":
                if layer.kernel_size != (1,1):
                    model._modules[name] = MixReGhos_Block(layer,[1,3,5])
                # else:
                #     model._modules[name] = MixReGhos_Block(layer, [1,1,1])
        else:
            add_ghostnet(layer, model_type)
    return model


class GroupedDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super(GroupedDepthwiseConv2d, self).__init__()

        self.in_channels = in_channels
        self.first = self.second = in_channels // 3
        self.third = in_channels - in_channels // 3 * 2
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.first, out_channels=self.first, kernel_size=kernel_sizes[0],
                      stride=stride, padding=kernel_sizes[0] // 2, groups=self.first, bias=False),
            nn.Conv2d(in_channels=self.second, out_channels=self.second, kernel_size=kernel_sizes[1],
                      stride=stride, padding=kernel_sizes[1] // 2, groups=self.second, bias=False),
            nn.Conv2d(in_channels=self.third, out_channels=self.third, kernel_size=kernel_sizes[2],
                      stride=stride, padding=kernel_sizes[2] // 2, groups=self.third, bias=False)

        ])
        nn.init.dirac_(self.depthwise_convs[0].weight, groups=self.first)
        nn.init.dirac_(self.depthwise_convs[1].weight, groups=self.second)
        nn.init.dirac_(self.depthwise_convs[2].weight, groups=self.third)

        # Batch Normalization（可选，根据需要添加或删除）
        # self.bns = nn.ModuleList([nn.BatchNorm2d(1) for _ in range(in_channels)])

    def forward(self, x):
        # 将输入数据按通道拆分
        # b,c,h,w = x.shape
        x_split = torch.split(x, [self.first,self.second,self.third], dim=1)
        # 对每个分组应用深度卷积
        out_list = [
            conv(x_part)  # 卷积
            for conv, x_part in zip(self.depthwise_convs, x_split)
        ]
        # 将结果沿通道方向拼接
        out = torch.cat(out_list, dim=1)

        return out

class MixReGhos_Block(nn.Module):
    def __init__(self, block, kernel_size=[1,3,5]):
        super(MixReGhos_Block, self).__init__()
        self.original_block = deepcopy(block)
        for param in self.original_block.parameters():
            param.requires_grad = False
        self.out_channels = self.original_block.out_channels
        self.Ghos_Mod = nn.Sequential(
            GroupedDepthwiseConv2d(self.out_channels,self.out_channels,kernel_sizes=kernel_size)
        )
        # nn.init.dirac_(self.Ghos_Mod[0].weight, groups=self.out_channels)

    def forward(self, x):
        output = self.Ghos_Mod(self.original_block(x))
        return output

log_dir = os.path.join(opt.save_dir, opt.base_task, opt.adapter_task)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()[0:10] + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
# result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
# utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


model_restoration = Restormer()


if opt.modify:
    state_dict = torch.load(opt.base_model, map_location='cuda:0')
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:] if 'module.' in k else k
        new_state_dict[name] = v
    model_restoration.load_state_dict(new_state_dict, strict=True)
    for param in model_restoration.parameters():
        param.requires_grad = False
    train_param = [param for param in model_restoration.parameters() if param.requires_grad]
    total_size_of_trainable_params = sum(param.numel() for param in train_param)
    model_restoration = add_ghostnet(model_restoration)
    if opt.adapter != "":
        checkpoint_extra = torch.load(opt.adapter)
        state_dict = checkpoint_extra["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model_restoration.load_state_dict(new_state_dict, strict=False)



train_param = [param for param in model_restoration.parameters() if param.requires_grad]
total_size_of_trainable_params = sum(param.numel() for param in train_param)
train_noparam = [param for param in model_restoration.parameters() if not param.requires_grad]
total_size_of_notrainable_params = sum(param.numel() for param in train_noparam)


with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(f"total_size_of_trainanble_params:{total_size_of_trainable_params}\n")
    f.write(f"total_size_of_notrainanble_params:{total_size_of_notrainable_params}\n")
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(train_param, lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(train_param, lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 200
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ###########
# base model resume
if opt.resume:
    path_chk_rest = opt.base_model
    print("Resume from " + path_chk_rest)
    # utils.load_checkpoint(model_restoration, path_chk_rest)
    checkpoint = torch.load(path_chk_rest)
    # try:
    #     model_restoration.load_state_dict(checkpoint["params"])
    # except:
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if 'module.' in k else k
    #     new_state_dict[name] = v
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    model_restoration.load_state_dict(new_state_dict)
    start_epoch = opt.checkpoint['epoch'] + 1

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


if opt.modify and opt.adapter!='':
    checkpoint_extra = torch.load(opt.adapter)
    start_epoch = checkpoint_extra["epoch"]

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=False, drop_last=False)

img_option_val = {'patch_size':opt.val_ps}
val_dataset = get_validation_data(opt.val_dir,img_option_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True,num_workers=0, drop_last=False)
len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)
###### validation ###########
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()

        factor = 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)
        restored = restored[:, :, :h, :w]
        input_ = input_[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1)
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
    psnr_dataset = sum(psnr_dataset) / len_valset
    psnr_model_init = sum(psnr_model_init) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
if opt.resume or opt.modify_resume:
    best_psnr = psnr_model_init
    best_epoch = start_epoch
else:
    best_psnr = 0
    best_epoch = 0

eval_now = len(train_loader) -1
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
            loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()

        #### Evaluation ####
        if i % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()

                    # =====
                    factor = 8
                    h,w = input_.shape[2],input_.shape[3]
                    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
                    # ======

                    filenames = data_val[2]
                    restored = model_restoration(input_)
                    restored = restored[:, :, :h, :w]
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict()
                                }, os.path.join(model_dir, f"model_best_{best_psnr}.pth"))
                    grad_parameters = OrderedDict(
                        (name, param) for name, param in model_restoration.named_parameters() if param.requires_grad)
                    # 保存这些参数
                    # torch.save(grad_parameters, 'ckpt/grad_params.pth')
                    torch.save({'state_dict': grad_parameters,'epoch':epoch}, os.path.join(model_dir, f"extra_best_{best_psnr}.pth"))
                print(
                    "[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d  Best_PSNR %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch,  best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d Best_PSNR %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch,  best_psnr) + '\n')
                torch.cuda.empty_cache()
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.12f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.12f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                }, os.path.join(model_dir, "model_latest.pth"))
    grad_parameters = OrderedDict(
        (name, param) for name, param in model_restoration.named_parameters() if param.requires_grad)
    torch.save({'state_dict': grad_parameters}, os.path.join(model_dir, "extra_latest.pth"))

    # if epoch % opt.checkpoint == 0:
    #     torch.save({'epoch': epoch,
    #                 'state_dict': model_restoration.state_dict(),
    #                 }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
    #     grad_parameters = OrderedDict(
    #         (name, param) for name, param in model_restoration.named_parameters() if param.requires_grad)
    #     torch.save({'state_dict': grad_parameters}, os.path.join(model_dir, "extra_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())
