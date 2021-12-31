import cv2
import sys
import numpy as np
import embedder as emb

class OpenCVEmbedder(emb.Embedder):
    def __init__(self, prototxt, caffemodel):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def do_inference(self, f):
        image = emb.read_image(f)

        self.net.setInput(image[np.newaxis])
        embedding = self.net.forward()

        embedding = embedding.flatten()
        np.savetxt(f.split('.')[0] + '_opencv_emb.txt', embedding, '%.6f')
        return embedding

def main():
    embedder = OpenCVEmbedder('resnetInception-512.prototxt', 'inception_resnet_v1_conv1x1.caffemodel')
    for f in sys.argv[1:]:
        embedding = embedder.do_inference(f)

if __name__ == '__main__':
    main()
