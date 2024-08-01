import torch
import torch.nn as nn
import numpy as np
import time
from rknn.api import RKNN

# Custom linear layer using supported MatMul and Add operations
class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

# Very simple encoder using supported operations
class SupportedEncoder(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(SupportedEncoder, self).__init__()
        self.linear1 = SimpleLinear(input_dim, model_dim)
        self.relu = nn.ReLU()
        self.linear2 = SimpleLinear(model_dim, model_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Very simple decoder using supported operations
class SupportedDecoder(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(SupportedDecoder, self).__init__()
        self.linear1 = SimpleLinear(input_dim, model_dim)
        self.relu = nn.ReLU()
        self.linear2 = SimpleLinear(model_dim, model_dim)

    def forward(self, x, memory):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def convert_to_rknn_onnx(model, input_size, rknn_model_path="supported_model.rknn", input_names=None):
    dummy_input = tuple(torch.randn(*size) for size in input_size)
    onnx_model_path = "supported_model.onnx"
    torch.onnx.export(model, dummy_input, onnx_model_path, input_names=input_names, opset_version=19)

    # Initialize RKNN
    rknn = RKNN()
    rknn.config(target_platform='rk3588')

    # Load the ONNX model into RKNN
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(onnx_model_path)
    if ret != 0:
        print('Load ONNX model failed!')
        exit(ret)

    print('Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build RKNN model failed!')
        exit(ret)

    print('Export RKNN model')
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)

    print('Done')
    return rknn

def measure_rknn_latency(rknn, input_data, num_iterations=10):
    latencies = []
    for _ in range(num_iterations):
        start_time = time.time()
        outputs = rknn.inference(inputs=input_data)
        latency = time.time() - start_time
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, outputs

if __name__ == "__main__":
    input_dim = 512  
    model_dim = 512  
    seq_length = 30  

    encoder = SupportedEncoder(input_dim, model_dim)
    decoder = SupportedDecoder(input_dim, model_dim)

    encoder_input_size = [1, seq_length, input_dim]
    decoder_input_size = [1, seq_length, input_dim]
    memory_size = [1, seq_length, input_dim]

    # Convert the encoder and decoder to RKNN
    rknn_encoder_path = 'supported_encoder.rknn'
    rknn_decoder_path = 'supported_decoder.rknn'
    rknn_encoder = convert_to_rknn_onnx(encoder, [encoder_input_size], rknn_encoder_path)
    rknn_decoder = convert_to_rknn_onnx(decoder, [decoder_input_size, memory_size], rknn_decoder_path, input_names=["input", "memory"])

    print('--> Initializing runtime environment')
    ret = rknn_encoder.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    
    ret = rknn_decoder.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('Runtime environment initialized')

    # Create dummy input data
    dummy_input = np.random.rand(1, seq_length, input_dim).astype(np.float32)
    memory_input = np.random.rand(1, seq_length, input_dim).astype(np.float32)

    # Measure the latency
    avg_encoder_latency, _ = measure_rknn_latency(rknn_encoder, [dummy_input])
    print(f"Average encoder latency: {avg_encoder_latency * 1000:.3f} ms")

    avg_decoder_latency, _ = measure_rknn_latency(rknn_decoder, [dummy_input, memory_input])
    print(f"Average decoder latency: {avg_decoder_latency * 1000:.3f} ms")
