import viz.graph

from settings import *

def show(args):
    inpath = os.path.join(DATA_DIR, args.model, "deploy.prototxt")
    outpath = os.path.join(OUTPUT_DIR, args.model+".eps")
    viz.graph.plot_graph(inpath, outpath)
    pass
