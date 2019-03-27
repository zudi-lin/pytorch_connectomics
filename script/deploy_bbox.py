import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage

import torch
import torch.nn as nn
import torch.utils.data
from libs import SynapseDataset, BboxSynapseDataset, collate_fn, collate_fn_bbox
from libs import fpn1
from libs.sync import DataParallelWithCallback

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
    D0 = args.output
    print(D0)
    sn = D0+'bbox_'+str(args.bbox)
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model_io_size, device, sn

# def get_logger(args):
#     log_name = args.output+'/log'
#     date = str(datetime.datetime.now()).split(' ')[0]
#     time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
#     log_name += '_approx_'+date+'_'+time
#     logger = open(log_name+'.txt','w') # unbuffered, write instantly

#     # tensorboardX
#     writer = SummaryWriter('visual/'+log_name)
#     return logger, writer    

def get_input(args, model_io_size, opt='val'):

    # load bbox dataset
    bbox_num = int(args.bbox)
    bbox_path = '/n/coxfs01/zudilin/research/simulation/data/v1_rand/'
    bbox_text = bbox_path + 'bb_1k.txt'
    #bbox_text = bbox_path + str(bbox_num) + '_20_bb.txt'
    print(bbox_text)
    assert(os.path.exists(bbox_text))
    g_text = None
    b_text = None 
    bbox_dataset = BboxSynapseDataset(bbox_text, g_text, b_text, data_aug=False, mode='test', use_gt=False)  
    bbox_loader = torch.utils.data.DataLoader(
            bbox_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_fn_bbox,
            num_workers=0, pin_memory=True)

    return bbox_loader

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def run_batch(args, volume, label, model, device, bbox, sn):
    volume = volume.to(device)
    output = model(volume)

    # empty cache
    output = output.detach().cpu() #(b, c, z, y, x)
    volume = volume.detach().cpu() #(b, z, y, x)
    #print(output.size()[0])
    torch.cuda.empty_cache()

    for k in range(len(bbox)):
        out_vol = (output[k].numpy()*255).astype(np.uint8)
        #out_vol[out_vol<32] = 0
        path = sn + '/' + '_'.join([str(x) for x in bbox[k]]) + '.h5'
        #print(path)
        writeh5(path, 'main', out_vol)

def validate(args, bbox_loader, model, device, sn):
    model.train()

    with torch.no_grad():
        for _, (volume, label, _, _, _, bbox) in enumerate(bbox_loader):
            #print(volume.size())
            run_batch(args, volume, label, model, device, bbox, sn)

    print('Finished: ', args.bbox)             

def main():
    args = get_args()

    print('0. initial setup')
    model_io_size, device, sn = init(args) 
    #_, _ = get_logger(args)
    print(sn)

    print('1. setup data')
    bbox_loader = get_input(args, model_io_size, 'val')

    print('2.0 setup model')
    #model = unet_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    #model = unet_NL_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    model = fpn1(in_num=1, out_num=3, filters=[32,64,128,256])
    print('model: ', model.__class__.__name__)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    model.load_state_dict(torch.load(args.pre_model))
    print('model weights path:')
    print(args.pre_model)
 
    print('4. start prediction')
    validate(args, bbox_loader, model, device, sn)
  
    print('5. finish validation')

if __name__ == "__main__":
    main()
