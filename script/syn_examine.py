from __future__ import division, print_function
import os, sys, glob
import numpy as np
import random
import pickle, h5py, time, argparse

# sample from bounding box
#from skimage.measure import label
from scipy import ndimage
    
def evaluate(filename):
    
    fl = h5py.File(filename, 'r')
    temp_pred = np.array(fl['main'])[:, 1:-1, 40:-40, 40:-40] / 255.0
    temp_pred = (temp_pred > 0.4).astype(np.float32)
    #print(temp_pred.shape)
    fl.close()

    pos_mask = temp_pred[0]
    neg_mask = temp_pred[1]
    syn_mask = temp_pred[2]

    pos_mask = pos_mask * syn_mask
    neg_mask = neg_mask * syn_mask

    # connected component
    pos_cc, num_pos = ndimage.label(pos_mask)
    neg_cc, num_neg = ndimage.label(neg_mask)
    print(pos_cc, num_neg)

    pos_region = np.unique(pos_cc)
    neg_region = np.unique(neg_cc)
    if len(pos_region) == 1: 
        pos_size = 0
    else:
        sizes = [(pos_cc==x).sum() for x in pos_region[1:]]
        pos_size = np.max(np.array(sizes))
    if len(neg_region) == 1:    
        neg_size = 0
    else:     
        sizes = [(neg_cc==x).sum() for x in neg_region[1:]]
        neg_size = np.max(np.array(sizes))

    # whether it's a synapse or not
    syn_true = (pos_size>100 and neg_size>100)

    name = filename.strip().split('.')[0]
    bbox = name.split('_')

    return bbox, syn_true

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-t','--input',  default='/n/coxfs01/',
                        help='Input folder')
    args = parser.parse_args()
    return args                    

if __name__== "__main__":   
    args = get_args()
    Prediction = []
    Box = []
    Do = args.input
    print(Do)       

    os.chdir(Do)
    file_list = glob.glob('*.h5')
    n_sample = len(file_list)
    print('number of samples:', n_sample)

    for i in range(n_sample):
        if i%100==0: print('finished: %d/%d' % (i, n_sample))
        bbox, syn_true = evaluate(file_list[i])
        Prediction.append(syn_true)
        Box.append(bbox) 

    with open('syn_bbox_loader.txt', 'w') as f:
        for k in range(len(Prediction)):
            f.write(','.join(list(Box[k])))
            f.write('\t%s\n' % int(Prediction[k]))   

    # good ratio
    ratio = (np.array(Prediction)).astype(int).sum() / len(Prediction)        
    print('good synapse ratio: ', ratio)
    print('All finished')        
