import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import sys
import struct
from glob import glob
import embedder as emb

def load_lfw_data(filepath):
    images = []
    files = glob(filepath + '/*/*.png')

    for f in files:
        images.append(emb.read_image(f))

    return np.ascontiguousarray(images, dtype=np.float32) # or .astype(np.float32)

class LFWEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size

        if not os.path.isfile(cache_file):
            # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
            self.data = load_lfw_data(training_data)
            self.current_index = 0
            # Allocate enough memory for a whole batch.
            self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class TensorRTEmbedder(emb.Embedder):
    def __init__(self, prototxt, caffemodel,
                 batch_size=1,
                 precision=trt.BuilderFlag.INT8,
                 output_tensor_name='flatten',
                 dataset_folder='lfw_mtcnnpy_160',
                 calib_cache_file='lfw_calibration.cache'):
        self.precision = precision
        self.engine = None
        self.tensor_scales = None
        self.logger = trt.Logger(trt.Logger.WARNING) #trt.Logger.VERBOSE)

        with trt.Builder(self.logger) as builder, \
             builder.create_network() as network, \
             builder.create_builder_config() as config, \
             trt.CaffeParser() as parser, \
             trt.Runtime(self.logger) as runtime:
            # We set the builder batch size to be the same as the calibrator's, as we use the same batches
            # during do_inferenceence. Note that this is not required in general, and inference batch size is
            # independent of calibration batch size.
            builder.max_batch_size = batch_size
            config.max_workspace_size = 1 << 30
            #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            config.set_flag(trt.BuilderFlag.DEBUG)
            if precision == trt.BuilderFlag.INT8:
                self.logger.log(trt.Logger.INFO, "=== INT8 Engine ===")
                config.set_flag(precision)
                config.int8_calibrator = LFWEntropyCalibrator(dataset_folder, cache_file=calib_cache_file)
            elif precision == trt.BuilderFlag.FP16:
                self.logger.log(trt.Logger.INFO, "=== FP16 Engine ===")
                config.set_flag(precision)
            else:
                self.logger.log(trt.Logger.INFO, "=== FP32 Engine ===")

            # Parse Caffe model
            model_tensors = parser.parse(deploy=prototxt, model=caffemodel, network=network, dtype=trt.float32)

            network.mark_output(model_tensors.find(output_tensor_name))

            if precision == trt.BuilderFlag.INT8:
                # Build engine only for calibration
                if not os.path.isfile(calib_cache_file):
                    builder.build_serialized_network(network, config)

                # Use int8 input/output tensor and int8 cuda kernel
                for layer in network:
                    for i in range(0, layer.num_outputs):
                        tensor = layer.get_output(i)
                        if tensor.name == output_tensor_name:
                            self.logger.log(trt.Logger.INFO, "change output tensor {}'s dtype from {} to {}".format(tensor.name, tensor.dtype, trt.int8))
                            #layer.precision = trt.int8
                            layer.set_output_type(i, trt.int8) # determine which tactic to use
                            tensor.dtype = trt.int8 # determine if reformat layer is needed

                input_tensor = network[0].get_input(0)
                self.logger.log(trt.Logger.INFO, "change input tensor {}'s dtype from {} to {}".format(input_tensor.name, input_tensor.dtype, trt.int8))
                input_tensor.dtype = trt.int8

            # Build engine
            plan = builder.build_serialized_network(network, config)
            self.engine = runtime.deserialize_cuda_engine(plan)

            if precision == trt.BuilderFlag.INT8:
                self.tensor_scales = self.parse_calibration_cache(calib_cache_file)

    def parse_calibration_cache(self, calib_cache_file):
        tensor_scales = {}
        with open(calib_cache_file) as f:
            for line in f:
                line = line.strip()
                fields = line.split(':')
                if len(fields) != 2:
                    continue

                layer_name = fields[0].strip()
                scale_hex = fields[1].strip()
                scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
                self.logger.log(trt.Logger.INFO, "{}: {}".format(layer_name, scale))

                tensor_scales[layer_name] = scale
        return tensor_scales

    def do_inference(self, f):
        engine = self.engine
        input_h = output_h = input_d = output_d = None
        bindings = []
        embedding = None

        assert engine.max_batch_size == 1

        for binding in engine:
            shape = engine.get_binding_shape(binding)
            size = trt.volume(shape) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                input_binding = binding
                self.logger.log(trt.Logger.INFO, "Input binding: '{}' of shape {} of type {}".format(binding, shape, dtype))
                input_h = host_mem
                input_d = device_mem
            else:
                output_binding = binding
                self.logger.log(trt.Logger.INFO, "Output binding: '{}' of shape {} of type {}".format(binding, shape, dtype))
                output_h = host_mem
                output_d = device_mem

        if self.precision == trt.BuilderFlag.INT8:
            image = np.rint(np.divide(emb.read_image(f), self.tensor_scales[input_binding])).astype(np.int8)
        else:
            image = emb.read_image(f)

        input_h[:] = image.reshape((-1))

        # Execute
        context = engine.create_execution_context()
        stream = cuda.Stream()

        cuda.memcpy_htod_async(input_d, input_h, stream)
        context.execute_async(batch_size=engine.max_batch_size, bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_h, output_d, stream)

        stream.synchronize()

        if self.precision == trt.BuilderFlag.INT8:
            np.savetxt(f.split('.')[0] + '_tensorrt_emb_int8.txt', output_h, '%d')
            embedding = np.multiply(output_h, self.tensor_scales[output_binding])
        else:
            embedding = output_h

        np.savetxt(f.split('.')[0] + '_tensorrt_emb.txt', embedding, '%.6f')
        return embedding

def main():
    embedder = TensorRTEmbedder('resnetInception-512.prototxt', 'inception_resnet_v1_conv1x1.caffemodel', precision=trt.BuilderFlag.INT8)
    for f in sys.argv[1:]:
        embedding = embedder.do_inference(f)

if __name__ == '__main__':
    main()
