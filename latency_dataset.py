# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

# TODO (to try):
# convert to onnx and then torch
# create a new script for the NPU and use a different approach of initializing encoders and decoders (like a different method or even library)
# try another paper (will be kinda hard depending on what I need to adapt for NPU)

import torch
import time
import pdb

import numpy as np

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from tqdm import tqdm

from rknn.api import RKNN
import sys

def export_model_to_onnx(model, dummy_input, onnx_path):
    try:
        print("Attempting to export model to ONNX...")
        model.eval()
        print(f"Model type: {type(model)}")
        print(f"Dummy input types: {[type(inp) for inp in dummy_input]}")
        print(f"Dummy input shapes: {[inp.shape for inp in dummy_input]}")
        
        # Additional prints to check the state right before export
        print("Preparing to call torch.onnx.export...")
        print(f"ONNX path: {onnx_path}")
        
        # Call to export the model
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            opset_version=19, 
            input_names=['src_tokens', 'src_lengths'], 
            output_names=['output']
        )
        
        print(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def convert_onnx_to_rknn(onnx_path, rknn_model_path):
    
    # create the rknn model object
    rknn = RKNN()
    
    # saving the fairseq model as a torchscript causes an error, so I have switched to converting to onnx and then rknn
    # script_model = torch.jit.script(model)
    # torch_model_path = rknn_model_path.replace('.rknn', '.pt')
    # script_model.save(torch_model_path)
    
    rknn.config(target_platform='rk3588') 
    
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX model failed!')
        exit(ret)
    print('done')   
    
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
    try:
        print("Attempting to measure RKNN latency...")
        latencies = []
        for _ in range(num_iterations):
            start_time = time.time()
            outputs = rknn.inference(inputs=input_data)
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        return avg_latency, outputs
    except Exception as e:
        print(f"Error measuring RKNN latency: {e}")
        sys.exit(1)

def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task
    task = tasks.setup_task(args)

    # Build model
    model = task.build_model(args)
    print(model)

    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]
    
    # Convert the encoder and decoder to RKNN
    onnx_encoder_path = 'encoder_model.onnx'
    onnx_decoder_path = 'decoder_model.onnx'
    
    if args.latnpu:
        try:
            print("Creating dummy tensors...")
            dummy_src_tokens_tensor = torch.tensor([dummy_src_tokens], dtype=torch.long)
            dummy_src_lengths_tensor = torch.tensor([dummy_sentence_length])
            dummy_inputs_encoder = (dummy_src_tokens_tensor, dummy_src_lengths_tensor)
            print(f"Dummy inputs for encoder created: {dummy_inputs_encoder}")

            dummy_prev_tensor = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
            dummy_inputs_decoder = (dummy_prev_tensor, dummy_src_lengths_tensor)
            print(f"Dummy inputs for decoder created: {dummy_inputs_decoder}")
        except Exception as e:
            print(f"Error creating dummy tensors: {e}")
            sys.exit(1)

        try:
            print("Exporting encoder to ONNX")
            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)
            model.set_sample_config(config_sam)  # Set the sample configuration before exporting

            export_model_to_onnx(model.encoder, dummy_inputs_encoder, onnx_encoder_path)
            print("Exporting decoder to ONNX")
            export_model_to_onnx(model.decoder, dummy_inputs_decoder, onnx_decoder_path)
        except Exception as e:
            print(f"Error exporting models to ONNX: {e}")
            sys.exit(1)

        try:
            print("Converting encoder ONNX to RKNN")
            rknn_encoder = convert_onnx_to_rknn(onnx_encoder_path, 'encoder_model.rknn')
            print("Converting decoder ONNX to RKNN")
            rknn_decoder = convert_onnx_to_rknn(onnx_decoder_path, 'decoder_model.rknn')
            
            print("Initializing encoder RKNN runtime")
            rknn_encoder.init_runtime(target=None) # change to 'rk3588' when run on the NPU
            print("Initializing decoder RKNN runtime")
            rknn_decoder.init_runtime(target=None)
        except Exception as e:
            print(f"Error converting ONNX to RKNN: {e}")
            sys.exit(1)
            
    # for latency predictor: latency dataset generation
    with open(args.lat_dataset_path, 'w') as fid:
        src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
        src_lengths_test = torch.tensor([dummy_sentence_length])
        prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)
        
        if args.latcpu:
            model.cpu()
            print('Measuring model latency on CPU for dataset generation...')
        elif args.latgpu:
            model.cuda()
            src_tokens_test = src_tokens_test.cuda()
            src_lengths_test = src_lengths_test.cuda()
            prev_output_tokens_test_with_beam = prev_output_tokens_test_with_beam.cuda()
            src_tokens_test.get_device()
            print('Measuring model latency on GPU for dataset generation...')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

        feature_info = utils.get_feature_info()
        fid.write(','.join(feature_info) + ',')
        latency_info = ['latency_mean_encoder', 'latency_mean_decoder', 'latency_std_encoder', 'latency_std_decoder']
        fid.write(','.join(latency_info) + '\n')

        for i in range(args.lat_dataset_size):
            print(i)
            config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)

            features = utils.get_config_features(config_sam)
            fid.write(','.join(map(str, features)) + ',')

            model.set_sample_config(config_sam)

            # dry runs
            for _ in range(5):
                encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

            encoder_latencies = []
            print('Measuring encoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()

                if not args.latnpu:
                    model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)

                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    encoder_latencies.append(start.elapsed_time(end))
                    if not args.latsilent:
                        print('Encoder one run on GPU (for dataset generation): ', start.elapsed_time(end))

                elif args.latcpu:
                    end = time.time()
                    encoder_latencies.append((end - start) * 1000)
                    if not args.latsilent:
                        print('Encoder one run on CPU (for dataset generation): ', (end - start) * 1000)
                
                # measuring the latency of the encoder on the NPU
                if args.latnpu:
                    encoder_latency, _ = measure_rknn_latency(rknn_encoder, [src_tokens_test.numpy(), src_lengths_test.numpy()], 1)
                    encoder_latencies.append(encoder_latency * 1000)

            # only use the 10% to 90% latencies to avoid outliers
            encoder_latencies.sort()
            encoder_latencies = encoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]
            print(f'Encoder latency for dataset generation: Mean: {np.mean(encoder_latencies)} ms; \t Std: {np.std(encoder_latencies)} ms')

            bsz = 1
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, args.beam).view(-1).long()
            if args.latgpu:
                new_order = new_order.cuda()

            encoder_out_test_with_beam = model.encoder.reorder_encoder_out(encoder_out_test, new_order)

            # dry runs
            for _ in range(5):
                model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam,
                                   encoder_out=encoder_out_test_with_beam)

            # decoder is more complicated because we need to deal with incremental states and auto regressive things
            decoder_iterations_dict = {'iwslt': 23, 'wmt': 30}
            if 'iwslt' in args.arch:
                decoder_iterations = decoder_iterations_dict['iwslt']
            elif 'wmt' in args.arch:
                decoder_iterations = decoder_iterations_dict['wmt']

            decoder_latencies = []
            print('Measuring decoder for dataset generation...')
            for _ in tqdm(range(args.latiter)):
                if args.latgpu:
                    start.record()
                elif args.latcpu:
                    start = time.time()
                
                if not args.latnpu:
                    incre_states = {}
                    for k_regressive in range(decoder_iterations):
                        model.decoder(prev_output_tokens=prev_output_tokens_test_with_beam[:, :k_regressive + 1],
                                       encoder_out=encoder_out_test_with_beam, incremental_state=incre_states)
                if args.latgpu:
                    end.record()
                    torch.cuda.synchronize()
                    decoder_latencies.append(start.elapsed_time(end))
                    if not args.latsilent:
                        print('Decoder one run on GPU (for dataset generation): ', start.elapsed_time(end))
                elif args.latcpu:
                    end = time.time()
                    decoder_latencies.append((end - start) * 1000)
                    if not args.latsilent:
                        print('Decoder one run on CPU (for dataset generation): ', (end - start) * 1000)
                elif args.latnpu:
                    # TODO this will need some fixes and testing in terms of incremental states and autoregressive components
                     
                    decoder_latency, _ = measure_rknn_latency(
                        rknn_decoder,
                        [prev_output_tokens_test_with_beam.numpy(), src_lengths_test.numpy()],  # Convert to numpy array
                        1  # Number of iterations is set to 1 because we are calling the function inside the loop
                    )
                    decoder_latencies.append(decoder_latency * 1000)

            # only use the 10% to 90% latencies to avoid outliers
            decoder_latencies.sort()
            decoder_latencies = decoder_latencies[int(args.latiter * 0.1): -max(1, int(args.latiter * 0.1))]

            print(decoder_latencies)
            print(f'Decoder latency for dataset generation: Mean: {np.mean(decoder_latencies)} ms; \t Std: {np.std(decoder_latencies)} ms')

            lats = [np.mean(encoder_latencies), np.mean(decoder_latencies), np.std(encoder_latencies), np.std(decoder_latencies)]
            fid.write(','.join(map(str, lats)) + '\n')

def cli_main():
    parser = options.get_training_parser()

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latnpu', action='store_true', help='measure SubTransformer latency on NPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/lat.tmp', help='the path to write latency dataset')
    parser.add_argument('--lat-dataset-size', type=int, default=200, help='number of data points for the dataset')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)

    if args.latcpu:
        args.cpu = True
        args.fp16 = False

    if args.pdb:
        pdb.set_trace()

    main(args)

if __name__ == '__main__':
    cli_main()
