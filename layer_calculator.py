from math import floor

import clarification
import torch

from clarification.models.clarification_dense import input_size_for_layer

sample_rate = 24000
dtype = torch.float32

sample_batch_ms = 300
samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)

overlap_ms = 5
overlap_samples = int((overlap_ms / 1000) * sample_rate)
dataset_batch_size = 16

device = "cpu"

target_size_error_range = 0.15

def simple_maker(name, layer_sizes):
    global sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

    model = clarification.models.ClarificationSimple(
        name=name,
        layer_sizes=layer_sizes,
        device=device, dtype=dtype)

    return model


def dense_maker(name, layer_sizes, invert=False, num_output_convblocks=2):
    global sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

    model = clarification.models.ClarificationDense(
        name=name,
        layer_sizes=layer_sizes, device=device, dtype=dtype, invert=invert,
        num_output_convblocks=num_output_convblocks)

    return model


# def dense_lstm_maker(name, layer_sizes, invert=False, num_output_convblocks=2):
#     global sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size
#
#     model = clarification.models.ClarificationDenseLSTM(
#         name=name,
#         in_channels=1,
#         layer_sizes=layer_sizes,
#         samples_per_batch=samples_per_batch,
#         device=device, dtype=dtype, invert=invert,
#         num_output_convblocks=num_output_convblocks)
#
#     return model


def resnet_maker(name, channel_size, layer_count):
    global sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

    model = clarification.models.ClarificationResNet(
        name=name,
        channel_size=channel_size,
        layer_count=layer_count,
        device=device, dtype=dtype)

    return model


def calculate_layers():

    target_size_to_good_dense_layers = {}
    target_size_to_good_dense_lstm_layers = {}
    target_size_to_good_simple_layers = {}

    target_sizes = []
    for i in range(20000, 100000, 20000):
        target_sizes.append(i)

    for i in range(100000, 500000, 50000):
        target_sizes.append(i)

    for i in range(600000, 1000000, 100000):
        target_sizes.append(i)

    for i in range(1500000, 5000000, 500000):
        target_sizes.append(i)

    layers_candidates = []
    for layer_count in range(3, 13, 2):
        print(".", end="")
        for base_layer_size in range(8, 384, 8):
            if base_layer_size <= 64:
                max_curve_step = 16
                curve_step_step = 8
            elif base_layer_size <= 128:
                max_curve_step = 32
                curve_step_step = 16
            elif base_layer_size <= 256:
                max_curve_step = 64
                curve_step_step = 16
            else:
                max_curve_step = 128
                curve_step_step = 16

            for curve_step in range(-max_curve_step, max_curve_step, curve_step_step):
                layers = []
                for layer_num in range(layer_count):
                    distance_from_middle = abs(layer_num - layer_count // 2)
                    layer_size = base_layer_size + distance_from_middle * curve_step
                    if layer_size <= 0:
                        break
                    layers.append(layer_size)

                if layers:
                    layers_candidates.append(layers)

    print(f"Evaluating {len(layers_candidates)} layers candidates")

    for target_size in target_sizes:
        target_size_min = target_size
        target_size_max = target_size * (1.0 + target_size_error_range)
        good_dense_layers = []
        good_dense_lstm_layers = []
        good_simple_layers = []

        print(f"Evaluating for parameters min: {target_size_min} max: {target_size_max}")
        for layers in layers_candidates:
            dense = dense_maker("dense", layers)
            dense_params = sum(p.numel() for p in dense.parameters())
            if target_size_max > dense_params > target_size_min:
                good_dense_layers.append((layers, dense_params))

            # dense_lstm = dense_lstm_maker("dense_lstm", layers)
            # dense_lstm_params = sum(p.numel() for p in dense_lstm.parameters())
            # if target_size_max > dense_lstm_params > target_size_min:
            #     good_dense_lstm_layers.append((layers, dense_lstm_params))

            simple_model = simple_maker("simple", layers)
            simple_params = sum(p.numel() for p in simple_model.parameters())

            if target_size_max > simple_params > target_size_min:
                good_simple_layers.append((layers, simple_params))

        target_size_to_good_dense_layers[target_size] = good_dense_layers
        target_size_to_good_dense_lstm_layers[target_size] = good_dense_lstm_layers
        target_size_to_good_simple_layers[target_size] = good_simple_layers

    target_size_to_good_resnet_layers = {ts: [] for ts in target_sizes}
    for channel_size in range(8, 384, 8):
        for layer_count in range(3, 12, 1):
            for target_size in target_sizes:
                target_size_min = target_size
                target_size_max = target_size * (1.0 + target_size_error_range)

                resnet = resnet_maker("resnet", channel_size, layer_count)
                resnet_params = sum(p.numel() for p in resnet.parameters())

                if target_size_max > resnet_params > target_size_min:
                    target_size_to_good_resnet_layers[target_size].append((channel_size, layer_count, resnet_params))

    for target_size in target_sizes:
        print(f"\n~~~~~~~~~~~~~~ Good layers for {target_size} ~~~~~~~~~~~~~~~~~~~~~~ \n")

        dense_layers = target_size_to_good_dense_layers[target_size]
        dense_lstm_layers = target_size_to_good_dense_lstm_layers[target_size]
        resnet_layers = target_size_to_good_resnet_layers[target_size]
        good_simple_layers = target_size_to_good_simple_layers[target_size]

        for layers, params in dense_layers:
            print(f"Dense Params: {params} Dense Layers: {layers}")

        for layers, params in dense_lstm_layers:
            print(f"Dense LSTM Params: {params} Dense LSTM Layers: {layers}")

        for channel_size, layer_count, params in resnet_layers:
            print(f"ResNet Params: {params} Channel Size: {channel_size} Layer Count: {layer_count}")

        for channel_size, params in good_simple_layers:
            print(f"Simple Params: {params} Channel Size: {channel_size}")


if __name__ == '__main__':
    calculate_layers()
