import os
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import numpy as np

from settings import *

from sets import Set

import json
import struct

import jinja2
templateLoader = jinja2.FileSystemLoader(searchpath = os.path.join(HL_TEMPLATES))
templateEnv    = jinja2.Environment(loader = templateLoader, extensions=['jinja2.ext.with_'])

class HLOperation(object):
    def __init__(self, name = None, type = None, input = None, params = None, bsize = None, psize = None):
        self.name = name
        self.type = type
        self.input = input
        self.params = params
        self.channels = 0
        self.bsize = bsize
        self.psize = psize

    def __str__(self):
        if self.input == None:
            iname = "None"
        else:
            iname = self.input[0].name
        return "%12s | %12s | %12s || %s" % (self.name, self.type, iname, self.params)


def generate_code(args):
    descr_path   = os.path.join(DATA_DIR, args.model, "deploy.prototxt")
    trained_path = os.path.join(DATA_DIR, args.model, args.model+".caffemodel")

    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(descr_path).read(), net_params)

    paramspath = os.path.join(DATA_DIR, args.model, "params")
    if not os.path.exists(paramspath):
        os.makedirs(paramspath)
    net = caffe.Net(descr_path, trained_path, caffe.TEST)

    print net_params.name

    renames  = {}
    nodes    = {}
    sequence = []

    for n in net_params.input:
        op = HLOperation(name = n, type = "Data" , bsize = net_params.input_dim)
        sequence.append(op)
        nodes[n] = op

    for layer in net_params.layer:
        name = layer.name.replace("-", "_")
        t = layer.type
        p = {}
        bottom = []
        bsize = None
        psize = None
        for l in layer.bottom:
            if l in renames.keys():
                bottom.append(nodes[renames[l]])
            else:
                bottom.append(nodes[l])
        if layer.type == "Convolution":
            params           = layer.convolution_param
            p['num_output']  = params.num_output
            p['pad']         = params.pad
            p['stride']      = params.stride
            p['kernel_size'] = params.kernel_size
            p['group']       = params.group
            p['bias_term']   = params.bias_term
            bsize = net.blobs[name].data.shape
            psize = net.params[name][0].data.shape

        elif layer.type == "ReLU":
            if layer.bottom[0] == layer.top[0]:
                name = layer.bottom[0]+"_relu"
                renames[layer.bottom[0]] = layer.bottom[0]+"_relu"
                bsize = nodes[layer.bottom[0]].bsize
        elif layer.type == "LRN":
            params = layer.lrn_param
            p['local_size'] = params.local_size
            p['alpha'] = params.alpha
            p['beta'] = params.beta
            bsize = net.blobs[name].data.shape
        elif layer.type == "Pooling":
            params = layer.pooling_param
            if params.pool == 0:
                t = "MaxPooling"
            p["kernel_size"] = params.kernel_size
            p["stride"] = params.stride
            bsize = net.blobs[name].data.shape
        elif layer.type == "InnerProduct":
            params = layer.inner_product_param
            p["num_output"] = params.num_output
            p['bias_term']   = params.bias_term
            bsize = net.blobs[name].data.shape
            psize = net.params[name][0].data.shape
        elif layer.type == "Dropout":
            continue
            params = layer.dropout_param
            p["dropout_ratio"] = params.dropout_ratio
        elif layer.type == "Softmax":
            bsize = net.blobs[name].data.shape
            pass
        op = HLOperation(name = name, type = t, input = bottom, params = p, bsize = bsize, psize = psize)
        sequence.append(op)
        nodes[name] = op

    print len(sequence), "steps"
    for op in sequence:
        print op

    datadir =os.path.join(DATA_DIR, args.model)
    if False:
        for k,v in net.params.items():
            if nodes[k].type == "Convolution" or nodes[k].type == "InnerProduct":
                print "conv params", k
                wpath = os.path.join(paramspath,"w_"+k+".npy")
                bpath = os.path.join(paramspath,"b_"+k+".npy")
                np.save(wpath, v[0].data)
                print v[0].data.shape
                if len(v) > 1:
                    np.save(bpath, v[1].data)

    main_template = templateEnv.get_template("hl_template.cpp.jinja")
    out = os.path.join(BASE_DIR,"src", "hl_%s.cpp" % args.model)
    with open(out, 'w') as f:
        f.write(main_template.render(sequence = sequence, nodes = nodes))

    test_template = templateEnv.get_template("test_conv.cpp.jinja")
    test = os.path.join(BASE_DIR,"test", "test_conv.cpp")

    outdir = os.path.join(OUTPUT_DIR,args.model)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(test, 'w') as f:
        f.write(test_template.render(sequence = sequence,
                                     nodes = nodes,
                                     datadir = datadir,
                                     outdir = outdir
                                     ))
