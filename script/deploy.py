from __future__ import division, print_function
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import h5py, yaml, json
from scipy.misc import imread
import itertools

# model
from libs import unet_SE_synBN
from libs import SynapseDataset, collate_fn_test
from libs.sync import DataParallelWithCallback

# get parameter
D0 = '/n/coxfs01/zudilin/research/synapseNet/data/jwr100/'
param = yaml.load(open(D0+'run_100-100-100.yaml'))
bfly_path = D0+"bfly_v2-2_add8.json"
bfly_db = json.load(open(bfly_path))
#param = yaml.load(open(D0+'run_20-20-20_v1.yaml'))

p_vol = param['data']
tile_sz = p_vol['tile-size']
z0=p_vol['z0'];z1=p_vol['z1']
y0=p_vol['y0'];y1=p_vol['y1']
x0=p_vol['x0'];x1=p_vol['x1']

# input volume: z,y,x
p_aff = param['aff']
in_sz = [p_aff['mz'], p_aff['my'], p_aff['mx']] # model output
vol_sz = [p_aff['vz'], p_aff['vy'], p_aff['vx']] # read more chunk
pad_sz = [p_aff['pz'], p_aff['py'], p_aff['px']] # read more chunk

# classifier
Dp = '/n/coxfs01/zudilin/research/synapseNet/outputs/jwr0829/' 
classifier_path = Dp + 'volume_1600000.pth'
destination = D0+"synapse/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blend(sz, opt=0):
    # Gaussian blending   
    zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                             np.linspace(-1,1,sz[1], dtype=np.float32),
                             np.linspace(-1,1,sz[2], dtype=np.float32), indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    sigma, mu = 0.6, 0.0
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    #print('weight shape:', ww.shape)

    return ww

# generate volume for prediction
def get_dataset(dataset_dict, x0, x1, y0, y1, z0, z1, tile_sz):
    # no padding at the boundary
    result = np.zeros((z1-z0, y1-y0, x1-x0), np.uint8)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = dataset_dict["sections"][z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+1, column=column+1)
                patch = imread(path, 0)
                xp0 = column * tile_sz
                xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def get_dest_path(x, y, z):
    return os.path.join(destination, str(x), str(y), str(z))

def process_message(classifier, x0, x1, y0, y1, z0, z1):
    p1 = get_dest_path(x0, y0, z0)
    redo = False
    path = os.path.join(p1, 'mask.h5')
    if not os.path.exists(path):
            redo = True

    if redo:
        #rh_logger.logger.report_event("process: %d,%d,%d" % (x0,y0,z0))
        aug_index = 0
        data = get_dataset(bfly_db, x0-pad_sz[2], x1+pad_sz[2], y0-pad_sz[1], y1+pad_sz[1], z0-pad_sz[0], z1+pad_sz[0], tile_sz)
        # print(np.max(data))
        # writeh5(os.path.join(p1, 'img.h5'), 'main', data)
        data = data / 255.0 # normalize image to (0,1)
        #print('data shape: ',data.shape)
        # out = [None] * 16 #different type of simple augmentation
        # for xflip, yflip, zflip, transpose in itertools.product(
        #                 (False, True), (False, True), (False, True), (False, True)):
        out = [None]
        for xflip, yflip, zflip, transpose in itertools.product(
                        (False, ), (False, ), (False, ), (False, )):                
                         
            print((aug_index, xflip, yflip, zflip, transpose))
            volume = data.copy() 
            if xflip:
                volume = volume[:, :, ::-1]
            if yflip:
                volume = volume[:, ::-1, :]
            if zflip:
                volume = volume[::-1, :, :]
            if transpose:
                volume = volume.transpose(0, 2, 1)
            # synapse: 3*z*y*x

            model_io_size = np.array(in_sz, dtype=int)
            #print('volume shape: ', volume.shape)
            #print('model_io_size: ', model_io_size)
            dataset = SynapseDataset(volume=[volume], label=None, vol_input_size=model_io_size,
                             vol_label_size=None, sample_stride=model_io_size/2,
                             data_aug=None, mode='test')

            test_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=28, shuffle=False, collate_fn = collate_fn_test,
                    num_workers=0, pin_memory=True)
            
            ww = blend(sz=model_io_size)
            sz = tuple([3]+list(model_io_size))
            result = np.zeros([3,z1-z0,y1-y0,x1-x0]).astype(np.float32)
            weight = np.zeros([z1-z0,y1-y0,x1-x0]).astype(np.float32)

            with torch.no_grad():
                for _, (pos, vol_input) in enumerate(test_loader):
                    vol_input = vol_input.to(device)
                    output = classifier(vol_input) # (b, 3, z, y, x)

                    for idx in range(output.size()[0]):
                        st = pos[idx]
                        result[:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += output[idx].cpu().detach().numpy() * np.expand_dims(ww, axis=0)
                        weight[st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww
            
            result = result / np.expand_dims(weight, axis=0)
            result = (result*255).astype(np.uint8)

            if transpose: # swap x-/y-affinity
                result = result.transpose(0, 1, 3, 2)
            if zflip:
                result = result[:, ::-1, :, :]
            if yflip:
                result = result[:, :, ::-1, :]
            if xflip:
                result = result[:, :, :, ::-1]
            out[aug_index] = result
            aug_index += 1

        if len(out) == 1:
            final = out[0]
        else:    
            final = np.mean(np.stack(out, axis=0), axis=0).astype(np.uint8)
        final[final < 128] = 0
        path = os.path.join(p1, 'mask.h5')
        writeh5(path, 'main', final)

def main_db(jobId, jobNum):
    #rh_logger.logger.start_process("Worker %d" % jobId, "starting", [])
    
    classifier = unet_SE_synBN(in_num=1, out_num=3, filters=[32,64,128,256], aniso_num=2)
    classifier = DataParallelWithCallback(classifier, range(2))
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(classifier_path))

    count = 0
    for x0a in range(x0, x1, vol_sz[2]):
        x1a = x0a + vol_sz[2]
        for y0a in range(y0, y1, vol_sz[1]):
            y1a = y0a + vol_sz[1]
            for z0a in range(z0, z1, vol_sz[0]):
                z1a = z0a + vol_sz[0]
                dir_path = get_dest_path(x0a, y0a, z0a)
                if count % jobNum == jobId:
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                        print('start prediction')
                        process_message(classifier, x0a, x1a, y0a, y1a, z0a, z1a)
                        print('prediction for volume %d is finished!\n' % (count))
                count += 1                        

def check_done():
    count = 0
    for x0a in range(x0, x1, vol_sz[2]):
        for y0a in range(y0, y1, vol_sz[1]):
            for z0a in range(z0, z1, vol_sz[0]):
                p1 = get_dest_path(x0a, y0a, z0a)
                count += 1
                path = os.path.join(p1, 'mask.h5')
                if not os.path.exists(path):
                    print("undone: %d, %d, %d" % (x0a, y0a, z0a))
    print('total jobs: ',count)
    
if __name__== "__main__":
    # python classify4-jwr_20um.py 0 0,1,2,3,4,5,6,7,9
    jobId = int(sys.argv[1])
    jobNum = int(sys.argv[2])
    main_db(jobId, jobNum) # single thread 
    #check_done() # single thread 