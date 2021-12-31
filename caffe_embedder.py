import sys
import os
os.environ['GLOG_minloglevel'] = '1' # suprress Caffe verbose prints
import caffe
import numpy as np
import embedder as emb

class CaffeEmbedder(emb.Embedder):
    def __init__(self, prototxt, caffemodel):
        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

    def do_inference(self, f):
        image = emb.read_image(f)

        self.net.blobs['data'].data[...] = image[np.newaxis]
        self.net.forward()

        embedding = self.net.blobs['flatten'].data.squeeze()
        np.savetxt(f.split('.')[0] + '_caffe_emb.txt', embedding, '%.6f')
        return embedding

def main():
    embedder = CaffeEmbedder('resnetInception-512.prototxt', 'inception_resnet_v1_conv1x1.caffemodel')
    for f in sys.argv[1:]:
        embedding = embedder.do_inference(f)

if __name__ == '__main__':
    main()
