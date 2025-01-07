from math import floor

import clarification
import torch

sample_rate = 24000
dtype = torch.float32

sample_batch_ms = 300
samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)

overlap_ms = 5
overlap_samples = int((overlap_ms / 1000) * sample_rate)
dataset_batch_size = 16

device = "cpu"

target_size_error_range = 0.15

def predict_clarification_dense_params(layer_sizes, num_output_convblocks=2):
    total_params = 0

    # First layer
    total_params += (layer_sizes[0] * 1 * 3 + layer_sizes[0]) * 2  # Conv1d + BatchNorm1d
    total_params += layer_sizes[0]  # Bias term for Conv1d

    # Down layers
    for i in range(len(layer_sizes) // 2):
        in_channels = sum(layer_sizes[:i+1])
        out_channels = layer_sizes[i + 1]
        total_params += (out_channels * in_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    # Up layers
    for i in range(len(layer_sizes) // 2 - 1):
        in_channels = sum(layer_sizes[:len(layer_sizes) // 2 + i + 1])
        out_channels = layer_sizes[len(layer_sizes) // 2 + i + 1]
        total_params += (out_channels * in_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    # Out layer
    in_channels = sum(layer_sizes)
    out_channels = layer_sizes[-1]
    total_params += (out_channels * in_channels * 2 + out_channels)  # ConvTranspose1d
    total_params += out_channels  # Bias term for ConvTranspose1d
    for _ in range(num_output_convblocks - 1):
        total_params += (out_channels * out_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    return total_params

def predict_clarification_dense_lstm_params(layer_sizes, num_output_convblocks=2):
    total_params = 0

    # First layer
    total_params += (layer_sizes[0] * 1 * 3 + layer_sizes[0]) * 2  # Conv1d + BatchNorm1d
    total_params += layer_sizes[0]  # Bias term for Conv1d

    # Down layers
    for i in range(len(layer_sizes) // 2):
        in_channels = sum(layer_sizes[:i+1])
        out_channels = layer_sizes[i + 1]
        total_params += (out_channels * in_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    # LSTM layer
    lstm_channel_size = 20
    hidden_size_multiplier = 4
    lstm_input_size = samples_per_batch // (2 ** (len(layer_sizes) // 2)) // 16
    total_params += lstm_channel_size * lstm_input_size * 4 * lstm_input_size  # LSTM weights
    total_params += lstm_channel_size * 4 * lstm_input_size  # LSTM biases

    # Up layers
    for i in range(len(layer_sizes) // 2 - 1):
        in_channels = sum(layer_sizes[:len(layer_sizes) // 2 + i + 1]) + lstm_channel_size
        out_channels = layer_sizes[len(layer_sizes) // 2 + i + 1]
        total_params += (out_channels * in_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    # Out layer
    in_channels = sum(layer_sizes) + lstm_channel_size
    out_channels = layer_sizes[-1]
    total_params += (out_channels * in_channels * 2 + out_channels)  # ConvTranspose1d
    total_params += out_channels  # Bias term for ConvTranspose1d
    for _ in range(num_output_convblocks - 1):
        total_params += (out_channels * out_channels * 3 + out_channels) * 2  # Conv1d + BatchNorm1d
        total_params += out_channels  # Bias term for Conv1d

    return total_params


def predict_resnet_params(channel_size, layer_count):
    """
    Estimates the number of parameters for a ClarificationResNet.

    Args:
      channel_size: The number of channels in the intermediate layers.
      layer_count: The number of residual blocks.

    Returns:
      The estimated number of parameters.
    """

    # Calculate parameters for the first layer
    first_layer_params = 1 * channel_size + channel_size  # conv1d + batchnorm1d

    # Calculate parameters for each ResBlock
    resblock_params = (
        channel_size * (channel_size // 2)  # conv1
        + (channel_size // 2)  # batchnorm2
        + (channel_size // 2) * (channel_size // 2) * 3  # conv4
        + (channel_size // 2)  # batchnorm5
        + (channel_size // 2) * channel_size  # conv7
    )

    # Calculate parameters for the last layer
    last_layer_params = (
        (channel_size + 1) * (channel_size + 1)  # conv1
        + (channel_size + 1)  # batchnorm1
        + (channel_size + 1) * 1  # conv2
    )

    # Total parameters
    total_params = (
        first_layer_params
        + layer_count * resblock_params
        + last_layer_params
    )

    total_params += channel_size * layer_count * 4

    return total_params


def predict_simple_params(layer_sizes):
    """
    Estimates the number of parameters for a ClarificationSimple model.

    Args:
      layer_sizes: A list of layer sizes.

    Returns:
      The estimated number of parameters.
    """
    total_params = 0

    # Calculate parameters for the first layer (ConvBlock1D)
    first_layer_params = (
        1 * layer_sizes[0]  # conv1d
        + layer_sizes[0]  # batchnorm1d
        + layer_sizes[0] * layer_sizes[0] * 3  # conv2
        + layer_sizes[0]  # batchnorm2
        + layer_sizes[0] * layer_sizes[0]  # conv3
    )
    total_params += first_layer_params

    # Calculate parameters for down layers
    for i in range(len(layer_sizes) // 2):
        in_channels = layer_sizes[i]
        out_channels = layer_sizes[i + 1]
        total_params += (
            in_channels * out_channels  # conv1
            + out_channels  # batchnorm1
            + out_channels * out_channels * 3  # conv2
            + out_channels  # batchnorm2
            + out_channels * out_channels  # conv3
        )

    # Calculate parameters for up layers
    layer_sizes_len = len(layer_sizes)
    for i in range(layer_sizes_len // 2 - 1):
        a_in_channels = layer_sizes[layer_sizes_len // 2 + i]
        out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
        total_params += (
            a_in_channels * out_channels  # conv1
            + out_channels  # batchnorm1
            + out_channels * out_channels * 3  # conv2
            + out_channels  # batchnorm2
            + out_channels * out_channels  # conv3
        )

    # Calculate parameters for the last layer (OutLayer)
    last_layer_in_channels = layer_sizes[-2]
    last_layer_out_channels = layer_sizes[-1]

    # ConvTranspose1d in OutLayer
    total_params += (
        last_layer_in_channels * last_layer_out_channels
    )

    # First ConvBlock1D in OutLayer
    total_params += (
        last_layer_out_channels * last_layer_out_channels  # conv1
        + last_layer_out_channels  # batchnorm1
        + last_layer_out_channels * last_layer_out_channels * 3  # conv2
        + last_layer_out_channels  # batchnorm2
        + last_layer_out_channels * last_layer_out_channels  # conv3
    )

    # Second ConvBlock1D in OutLayer (last_layer=True)
    total_params += (
        last_layer_out_channels * last_layer_out_channels  # conv1
        + last_layer_out_channels  # batchnorm1
        + last_layer_out_channels * 1  # conv2
    )

    if len(layer_sizes) < 4:
        total_params *= 1.2
    elif len(layer_sizes) < 6:
        total_params *= 1.3
    elif len(layer_sizes) < 8:
        total_params *= 1.4
    else:
        total_params *= 1.5

    total_params = floor(total_params)

    return total_params

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
        in_channels=1,
        layer_sizes=layer_sizes, device=device, dtype=dtype, invert=invert,
        num_output_convblocks=num_output_convblocks)

    return model


def dense_lstm_maker(name, layer_sizes, invert=False, num_output_convblocks=2):
    global sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

    model = clarification.models.ClarificationDenseLSTM(
        name=name,
        in_channels=1,
        layer_sizes=layer_sizes,
        samples_per_batch=samples_per_batch,
        device=device, dtype=dtype, invert=invert,
        num_output_convblocks=num_output_convblocks)

    return model


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
            # dense = dense_maker("dense", layers)
            # dense_params = sum(p.numel() for p in dense.parameters())

            # dense_params_prediction = predict_clarification_dense_params(layers)
            # dense_diff = abs(dense_params - dense_params_prediction)
            # dense_diff_percentage = dense_diff / dense_params
            # print(f"Dense Predicted: {dense_params_prediction} Actual: {dense_params} Diff: {dense_diff} Diff %: {dense_diff_percentage}")

            dense_params = predict_clarification_dense_params(layers)

            if target_size_max > dense_params > target_size_min:
                good_dense_layers.append((layers, dense_params))

            # dense_lstm = dense_lstm_maker("dense_lstm", layers)
            # dense_lstm_params = sum(p.numel() for p in dense_lstm.parameters())

            # dense_lstm_params_prediction = predict_clarification_dense_lstm_params(layers)
            # dense_lstm_diff = abs(dense_lstm_params - dense_lstm_params_prediction)
            # dense_lstm_diff_percentage = dense_lstm_diff / dense_lstm_params
            # print(f"Dense Predicted: {dense_lstm_params_prediction} Actual: {dense_lstm_params} Diff: {dense_lstm_diff} Diff %: {dense_lstm_diff_percentage}")

            dense_lstm_params = predict_clarification_dense_lstm_params(layers)

            if target_size_max > dense_lstm_params > target_size_min:
                good_dense_lstm_layers.append((layers, dense_lstm_params))

            # simple_model = simple_maker("simple", layers)
            # simple_params = sum(p.numel() for p in simple_model.parameters())
            #
            # simple_params_predictio n = predict_simple_params(layers)
            # simple_diff = abs(simple_params - simple_params_prediction)
            # simple_diff_percentage = simple_diff / simple_params * 100
            # print(f"Simple layer sizes:{layers} Predicted: {simple_params_prediction} Actual: {simple_params} Diff: {simple_diff} \t\tDiff %: {simple_diff_percentage}")

            simple_params = predict_simple_params(layers)

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

                # resnet = resnet_maker("resnet", channel_size, layer_count)
                # resnet_params = sum(p.numel() for p in resnet.parameters())

                # resnet_params_prediction = predict_resnet_params(channel_size, layer_count)
                # resnet_diff = abs(resnet_params - resnet_params_prediction)
                # resnet_diff_percentage = resnet_diff / resnet_params * 100
                # print(f"ResNet Predicted: {resnet_params_prediction} Actual: {resnet_params} Diff: {resnet_diff} Diff %: {resnet_diff_percentage}")
                # print(f"channel_size: {channel_size} layer_count: {layer_count} diff: {resnet_diff}")

                resnet_params = predict_resnet_params(channel_size, layer_count)

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
