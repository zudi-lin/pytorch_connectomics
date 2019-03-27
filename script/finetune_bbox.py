import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage
import random

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from libs import BboxSynapseDataset, PolaritySynapseDataset, collate_fn, collate_fn_bbox
from libs import WeightedMSE
from libs import fpn1
from libs.sync import DataParallelWithCallback

# from dataset import SynapseDataset, collate_fn
# from loss import WeightedBCELoss
# from model import res_unet

# tensorboardX
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('--bbox', default=None)
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    # parser.add_argument('-v','--val',  default='',
    #                     help='input folder (test)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                        help='Ground-truth label path')
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='I/O size of deep network')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')                    
    parser.add_argument('-ft','--finetune', type=bool, default=False,
                        help='Fine-tune on previous model [Default: False]')
    parser.add_argument('-pm','--pre-model', type=str, default='',
                        help='Pre-trained model path')                  

    # optimization option
    parser.add_argument('-lt', '--loss', type=int, default=1,
                        help='Loss function')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate')
    # parser.add_argument('-lr_decay', default='inv,0.0001,0.75',
    #                     help='learning rate decay')
    # parser.add_argument('-betas', default='0.99,0.999',
    #                     help='beta for adam')
    # parser.add_argument('-wd', type=float, default=5e-6,
    #                     help='weight decay')
    parser.add_argument('--volume-total', type=int, default=1000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    # training settings:
    parser.add_argument('-dw','--distance', default='im_uint8.h5',
                        help='Loss weight based on distance transform.')
    args = parser.parse_args()
    return args

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_io_size, device

def get_input(args, model_io_size, opt='train'):
    # two dataLoader, can't be both multiple-cpu (pytorch issue)

    data_path = '/n/coxfs01/zudilin/research/data/JWR/syn_vol1/'
    img_name = [data_path + 'im.h5']
    seg_name = [data_path + 'jwr_syn_polarity.h5']

    train_input = [None]
    train_label = [None]
    # original image is in [0, 255], normalize to [0, 1]
    for i in range(len(img_name)):
        train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        train_label[i] = (train_label[i] != 0).astype(np.float32)
        train_input[i] = train_input[i].astype(np.float32)
    
        assert train_input[i].shape==train_label[i].shape[1:]
        print("synapse pixels: ", np.sum(train_label[i][0]))
        print("volume shape: ", train_input[i].shape)    

    data_aug = True
    SHUFFLE = (opt=='train')
    print('Data augmentation: ', data_aug)
    dataset = PolaritySynapseDataset(volume=train_input, label=train_label, vol_input_size=model_io_size,
                                 vol_label_size=model_io_size, data_aug = data_aug, mode = 'train')                            

    print('Batch size: ', args.batch_size)
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
            num_workers=args.num_cpu, pin_memory=True)

    bbox_path = '/n/coxfs01/zudilin/research/simulation/data/v1_rand/'
    bbox_text = bbox_path + 'bb.txt'
    g_text = bbox_path + 'pseudo_label/random_1280_g.txt' 
    b_text = bbox_path + 'pseudo_label/random_1280_b.txt' 
    print(bbox_path)
    print(g_text)
    print(b_text)

    bbox_dataset = BboxSynapseDataset(bbox_text, g_text, b_text, data_aug=True, mode='train', use_gt=True)  
    bbox_loader = torch.utils.data.DataLoader(
            bbox_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn_bbox,
            num_workers=args.num_cpu, pin_memory=True)

    return img_loader, bbox_loader

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def run_stat(args, volume_id):

    Do = '/n/coxfs01/zudilin/research/simulation/'
    log_path = Do + 'log/' + args.output+'/'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    def get_pref(mem=20000, do_gpu=False):
        pref = '#!/bin/bash\n'
        pref+= '#SBATCH -N 1 # number of nodes\n'
        pref+= '#SBATCH -p cox\n'
        pref+= '#SBATCH -n 4 # number of cores\n'
        pref+= '#SBATCH --mem '+str(mem)+' # memory pool for all cores\n'
        if do_gpu:
            pref+= '#SBATCH --gres=gpu:2 # memory pool for all cores\n'
        pref+= '#SBATCH -t 12:00:00 # time (D-HH:MM)\n'
        pref+= '#SBATCH -o logs/deploy_%j.log\n\n'
        pref+= 'module load cuda\n'
        pref+= 'source activate py3_pytorch\n\n'
        return pref

    mem=50000
    do_gpu= True

    cmd1 = 'a=%d' % (volume_id)
    cmd2 = 'b=%s' % (os.path.basename(args.output+'/'))

    # 1. Deploy on bbox
    cmd3 = 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -u deploy_bbox.py -mi 4,128,128 -g 2 -c 4 -b 64 \ \n -o bbox_result/${b}/ -pm outputs/${b}/volume_${a}.pth --bbox ${a}\n\n'

    # 2. Generate label
    cmd4 = 'python -u syn_examine_mul.py -t bbox_result/${b}/bbox_${a}/\n\n'

    # 3. Run stats
    cmd5 = 'python -u stat.py -t bbox_result/${b}/bbox_${a}/\n'

    pref=get_pref(mem, do_gpu)

    if not os.path.exists(Do):
        os.makedirs(Do)

    num=20
    fn = 'deploy'
    for b in range(num):
        fl=open(Do + fn+'_%d.sh'%(b),'w')
        fl.write(pref)
        fl.write('b=%d\n\n'%(b))
        fl.write(cmd1)
        fl.write(cmd2)
        fl.close()

    # run test job
    os.system('sbatch deploy_%d.sh' % (volume_id))

def train_batch(args, volume, label, class_weight, model, device, criterion, optimizer, logger, writer, volume_id):
    volume, label = volume.to(device), label.to(device)
    class_weight = class_weight.to(device)
    output = model(volume)

    loss = criterion(output, label, class_weight)
    writer.add_scalar('MSE Loss', loss.item(), volume_id) 

    # compute gradient and do Adam step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
            loss.item(), optimizer.param_groups[0]['lr']))

    # LR update
    #if args.lr > 0:
        #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
    # print('volume_id: ', volume_id)
    if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
        torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))
        print('save model!')
        print('run evaluation!')
    # Terminate
    if volume_id >= args.volume_total:
        print('finished!')   #
        exit(0)

    return output    

def train(args, train_loader, bbox_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    #import random
    model.train()
    volume_id = 0

    data_loader = iter(train_loader)
    while True:
        # for _, (volume, label, class_weight, _) in enumerate(bbox_loader):
        for _, (volume, label, class_weight, _, gt, _) in enumerate(bbox_loader):
            volume_id += args.batch_size
            output = train_batch(args, volume, label, class_weight, model, device, criterion, optimizer, logger, writer, volume_id)

            if volume_id % (10*args.batch_size) == 0:
                visualize(volume, label, output, volume_id, writer, gt)
            if random.random() > 0.5:
                volume, label, class_weight, _ = next(data_loader)
                volume_id += args.batch_size
                train_batch(args, volume, label, class_weight, model, device, criterion, optimizer, logger, writer, volume_id)


def visualize(volume, label, output, iteration, writer, gt):
    ###
    sz = volume.size() #[8, 1, 8, 160, 160]
    #print('pseudo-label: ', gt[0])
    # if iteration==0:
    #     print(volume.size())
    #     print(output.size())
    #     print(label.size())

    volume_visual = volume[0].detach().cpu().squeeze().unsqueeze(1).expand(sz[2],3,sz[3],sz[4])
    output_visual = output[0].detach().cpu().squeeze().permute(1,0,2,3)
    label_visual = label[0].detach().cpu().squeeze().permute(1,0,2,3)

    # if iteration==0:
    #     print(volume_visual.size())
    #     print(output_visual.size())
    #     print(label_visual.size())

    volume_show = vutils.make_grid(volume_visual, nrow=8, normalize=True, scale_each=True)
    output_show = vutils.make_grid(output_visual, nrow=8, normalize=True, scale_each=True)
    label_show = vutils.make_grid(label_visual, nrow=8, normalize=True, scale_each=True)

    writer.add_image('Input', volume_show, iteration)
    writer.add_image('Label', label_show, iteration)
    writer.add_image('Output', output_show, iteration)   
    

def main():
    args = get_args()

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader, bbox_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    #model = unet_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    #model = unet_NL_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    model = fpn1(in_num=1, out_num=3, filters=[32,64,128,256])
    print('model: ', model.__class__.__name__)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    #from torchsummary import summary
    #summary(model, input_size=(1, 8, 160, 160))

    print('Fine-tune? ', bool(args.finetune))
    if bool(args.finetune):
        model.load_state_dict(torch.load(args.pre_model))
        print('fine-tune on previous model:')
        print(args.pre_model)
            
    print('2.1 setup loss function')
    criterion = WeightedMSE()   
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-5, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)

    print('4. start training')
    train(args, train_loader, bbox_loader, model, device, criterion, optimizer, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
