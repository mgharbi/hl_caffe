import os
import caffe
import time
from settings import *
import numpy as np

def run(args):
    descr_path   = os.path.join(DATA_DIR, args.model, "deploy.prototxt")
    trained_path = os.path.join(DATA_DIR, args.model, args.model+".caffemodel")
    if args.use_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
    start = time.time()
    net = caffe.Net(descr_path, trained_path, caffe.TEST)
    end = time.time()

    # I = np.random.random((223,223,10))
    # I = I[np.newaxis,:,:,:]
    # I = I.transpose((0,3,1,2))
    # net.blobs['data'].reshape(*I.shape)
    # net.blobs['data'].data[...] = I

    print "Blobs:"
    for k,v in net.blobs.items():
        print "%30s |" % k, v.data.shape
    print "\n"
    print "Parameters:"
    n_params = 0
    for k,v in net.params.items():
        sh = v[0].data.shape
        print "%30s |" % k, sh
        n_params += np.prod(sh)
    print "%d parameters." % n_params
    print "\n"

    print "Set net %s, %f ms elapsesd" % (args.model, (end-start)*1000)
    start = time.time()
    nruns = 10
    for i in range(nruns):
        net.forward()
    end = time.time()
    print "Run net %s, %.2f ms elapsesd" % (args.model, (end-start)*1000/nruns)
