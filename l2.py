#!/usr/bin/env python3

import sys
import numpy as np
import pandas
import embedder as emb
from caffe_embedder import CaffeEmbedder
from opencv_embedder import OpenCVEmbedder
from tensorrt_embedder import TensorRTEmbedder

np.set_printoptions(suppress=True)

def main():
    prototxt = 'resnetInception-512.prototxt'
    caffemodel = 'inception_resnet_v1_conv1x1.caffemodel'
    embedder = CaffeEmbedder(prototxt, caffemodel)
    # embedder = OpenCVEmbedder(prototxt, caffemodel)
    # embedder = TensorRTEmbedder(prototxt, caffemodel)
    embeddings = {}

    inputs = sys.argv[1:]

    for f in inputs:
        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.ppm'):
            embedding = emb.l2_normalize(embedder.do_inference(f))
        elif f.endswith('.dimg'):
            embedding = emb.l2_normalize(np.multiply(np.fromfile(f, sep=' '), 0.034820929169654846))
        elif f.endswith('.txt'):
            embedding = emb.l2_normalize(np.fromfile(f, sep=' \n'))
        else:
            print('Unsupported file type: {}'.format(f))
            sys.exit()

        embeddings[f] = embedding

    distances = np.empty((len(inputs), len(inputs)))

    for f1 in inputs:
        for f2 in inputs:
            distances[inputs.index(f1)][inputs.index(f2)] = emb.l2_distance(embeddings[f1], embeddings[f2])

    print(pandas.DataFrame(distances, inputs, inputs))

if __name__ == '__main__':
    main()
