import os
import caffe
import time
from settings import *
import numpy as np
from PIL import Image

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

    # I = np.random.random((223,223,3))
    I = np.array(Image.open(os.path.join(DATA_DIR, args.model,"input.png"))).astype(np.float32)/255
    I = I[np.newaxis,:,:,:]
    I = I.transpose((0,3,1,2))
    # net.blobs['data'].reshape(*I.shape)
    net.blobs['data'].data[...] = I

    print "Set net %s, %f ms elapsesd" % (args.model, (end-start)*1000)
    start = time.time()
    nruns = 10
    for i in range(nruns):
        net.forward()
    end = time.time()
    print "Run net %s, %.2f ms elapsesd" % (args.model, (end-start)*1000/nruns)


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

    # d = net.blobs['conv2'].data
    # d = d.transpose((2,3,1,0))
    # d = d[:,:,:,0]
    # s = ""
    # for y in range(d.shape[0]):
    #     for x in range(10):
    #         s+= "%.5f " % d[y,x,0]
    #     s+= "\n"
    # print s

    d = net.blobs['prob'].data
    d = d[0,:]
    if len(d.shape) == 3:
        print d.shape
        # d = d.transpose((1,2,0))
    # d = np.ravel(d)
    # s = ""
    # for x in range(d.shape[0]):
    #     s+= "%.5f " % d[x]
    #     if (x+1) % 5 == 0 and x >0:
    #         s+= "\n"
    # print s

    cpp = np.load(os.path.join(OUTPUT_DIR, args.model, "output.npy"))
    print cpp.shape
    # cpp  = np.ravel(cpp)

    print "cpp:",  np.amax(np.abs(cpp))
    print "caffe:",  np.amax(np.abs(d))
    err = np.abs(cpp-d)
    print np.amax(err), "at", np.unravel_index(np.argmax(err),err.shape)
