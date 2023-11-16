import copy
import time

import numpy as np
import torchvision
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG11_Weights
import torch

from MyDataset import ImageNetDataset, IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH
from plot import plot
from quality_evaluation import quantize, evaluate

model_fp32 = torchvision.models.vgg11(weights='IMAGENET1K_V1').eval()
transforms = VGG11_Weights.IMAGENET1K_V1.transforms()


def measure_single_conv():
    def quantized_single_conv_forward(qmodel, x):
        features_0_input_scale_0 = qmodel.features_0_input_scale_0
        features_0_input_zero_point_0 = qmodel.features_0_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, features_0_input_scale_0, features_0_input_zero_point_0, torch.quint8);
        x = features_0_input_scale_0 = features_0_input_zero_point_0 = None
        features_0 = getattr(qmodel.features, "0")(quantize_per_tensor);
        quantize_per_tensor = None
        features_2 = getattr(qmodel.features, "2")(features_0);
        features_0 = None
        features_3 = getattr(qmodel.features, "3")(features_2);
        features_2 = None
        features_5 = getattr(qmodel.features, "5")(features_3);
        features_3 = None
        features_6 = getattr(qmodel.features, "6")(features_5);
        features_5 = None
        start = time.time()
        _ = getattr(qmodel.features, "8")(features_6);
        features_6 = None
        return time.time() - start

    backbone_fp32 = copy.deepcopy(model_fp32)
    backbone_fp32.avgpool = nn.Identity()
    backbone_fp32.classifier = nn.Identity()
    main_conv = backbone_fp32.features[8]
    for i in range(8, 21):
        backbone_fp32.features[i] = nn.Identity()

    def f32_forward(x):
        return backbone_fp32.features(x)

    backbone_fp32.forward = f32_forward

    batch_size = 1
    n_samples = 1000
    dataset = ImageNetDataset(IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH, transforms,
                              pre_load=True, dataset_size=n_samples * batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    quantization_types = ["x86", "fbgemm", "onednn"]  # "none", "qnnpack",
    all_times = []

    times = []
    for i, (tensor, class_id) in enumerate(tqdm.tqdm(dataloader)):
        f32_input = backbone_fp32(tensor)
        start = time.time()
        _ = main_conv(f32_input)
        times.append(time.time() - start)
    times = sorted(times)[:int(0.95 * n_samples)]
    print("fp32", np.mean(times), np.median(times))
    all_times.append(times)

    for q_type in quantization_types:
        torch.backends.quantized.engine = q_type
        quantized_model = quantize(model_fp32, q_type, transforms=transforms)

        qtimes = []
        for i, (tensor, class_id) in enumerate(tqdm.tqdm(dataloader)):
            qtime = quantized_single_conv_forward(quantized_model, tensor)
            qtimes.append(qtime)
        qtimes = sorted(qtimes)[:int(0.95 * n_samples)]
        print(q_type, np.mean(qtimes), np.median(qtimes))
        all_times.append(qtimes)

    plot(all_times, ["fp32"] + quantization_types, f"i9_single_conv_filtered_{n_samples}_{batch_size}")


def inner_loop(n_samples, batch_size, cpu_type):
    all_times = []
    quantization_types = ["x86", "fbgemm", "onednn"] # "none", "qnnpack",
    for q_type in quantization_types:
        if q_type != 'none':
            torch.backends.quantized.engine = q_type
            quantized_model = quantize(model_fp32, q_type, transforms=transforms)
        else:
            quantized_model = model_fp32
        print("evaluating")
        accuracy, current_times = evaluate(quantized_model,
                                           transforms,
                                           n_samples=n_samples,
                                           batch_size=batch_size,
                                           pre_load_dataset=True,
                                           return_time=True)
        current_times = sorted(current_times)[:int(0.95 * n_samples)]
        all_times.append(current_times)
        print("!!!", n_samples, batch_size)
        print(q_type, accuracy, np.mean(current_times), np.median(current_times))
    plot(all_times, quantization_types, f"{cpu_type}_inner_loop_{n_samples}_{batch_size}")


def outer_loop(n_samples, batch_size, cpu_type):
    quantization_types = ["none", "x86", "fbgemm", "onednn"]  # "qnnpack",
    quantized_models = []
    for q_type in quantization_types:
        if q_type != 'none':
            torch.backends.quantized.engine = q_type
            quantized_model = quantize(model_fp32, q_type, transforms=transforms)
        else:
            quantized_model = model_fp32
        quantized_models.append(quantized_model)

    dataset = ImageNetDataset(IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH, transforms,
                              pre_load=True, dataset_size=n_samples * batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    times = []
    for i, (tensor, class_id) in enumerate(tqdm.tqdm(dataloader)):
        sample_times = []
        for model, q_type in zip(quantized_models, quantization_types):
            torch.backends.quantized.engine = q_type

            start = time.time()
            _ = model(tensor)
            sample_times.append(time.time() - start)
        times.append(sample_times)
    times = np.array(times).T.tolist()
    print(quantization_types, np.mean(times, axis=1), np.median(times, axis=1))
    plot(times, quantization_types, f"{cpu_type}_outer_loop_{n_samples}_{batch_size}")


if __name__ == "__main__":
    # inner_loop(2000, 1, "i9")
    # inner_loop(1000, 2, "i9")
    # inner_loop(500, 4, "i9")
    # inner_loop(400, 8, "i9")
    # inner_loop(300, 16, "i9")
    # inner_loop(200, 32, "i9")

    # outer_loop(2000, 1, "i9")

    measure_single_conv()

