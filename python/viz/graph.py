import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def plot_graph(modelpath, outpath):
    net_params = caffe_pb2.NetParameter()
    text_format.Merge(open(modelpath).read(), net_params)
    caffe.draw.draw_net_to_file(net_params, outpath)

