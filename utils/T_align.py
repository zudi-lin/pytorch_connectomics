import os,sys
import numpy as np
import h5py

from T_util import *

opt=sys.argv[1]
if opt[0] == '0': # 300x300 image around the synapse
    if opt == '0': # 300x300 image around the synapse
        # for i in {0..5};do python T_jwr_syn.py 0 ${i} 6 & done
        # simple iterate
        # sa idm 
        import json
        from scipy.misc import imsave

        jobId = int(sys.argv[2])
        jobNum = int(sys.argv[3])
        
        dopt=1 
        if dopt == 0:
            # cvpr 19: syn
            D0 = '/n/coxfs01/donglai/ppl/zudi/jwr/syn_test_0904'
            Dim = D0+'_im/'
            Db = D0+'_wdj/v1_rand/'
            Do=Dim
            Dh = '/n/coxfs01/zudilin/research/synapseNet/data/jwr100/simulation/bbox_result/bbox_1600/' 
            psz = [3, 8, 160, 160]
        elif dopt == 1:
            # iccv 19
            D0 = '/n/coxfs01/donglai/ppl/zudi/jwr/syn_test_0322'
            Dim = D0+'_im/'
            Db = D0+'_wdj/v1_rand/'
            Do=Dim
            Dh = '/n/coxfs01/zudilin/research/simulation/bbox_result/ours_1_all/bbox_12800/'
            psz = [3, 4, 128, 128]

        syn_bb = readtxt(Db+'bb.txt')
        fn= readtxt(Db+'fn.txt')
        l1 = [x[:-1].replace(',','_') for x in syn_bb]
        did = np.arange(len(syn_bb))

        for dd in did[jobId::jobNum]:
            # print xi,yi,zi,b
            sn = fn[dd][fn[dd].find('/'):-1]
            s1 = Dim+sn
            s2 = Do+sn
            # load synapse
            data = np.array(h5py.File(Dh+l1[dd]+'.h5','r')['main'][:,psz[1]//2])
            p1 = 128*(data[0]>128)
            p2 = 255*(data[1]>128)
            if np.count_nonzero(p1)>np.count_nonzero(p1):
                syn=p1;syn[p2>0] = p2[p2>0]
            else:
                syn=p2;syn[p1>0] = p1[p1>0]
            # load rotation
            pp = np.loadtxt(s1+'_rot.txt')
            # re-center: 300 -> 160
            imsave(s2+'_syn2_r1.png', rotateIm(syn.astype(np.uint8), pp[0], (pp[1]-70,pp[2]-70)))
    elif opt =='0.1': # check done
        D0 = '/n/coxfs01/donglai/ppl/zudi/jwr/syn_test_0904'
        Dim = D0+'/../'
        Db = D0+'_wdj/v1_rand/'
        fn= readtxt(Db+'fn.txt')
        suf='_syn2_r1'
        st  = int(sys.argv[2])
        fn = fn[st:]
        for ii,i in enumerate(fn):
            sn = Dim+i[:-1]+suf+'.png'
            print st+ii,sn
            if not os.path.exists(sn):
                break
