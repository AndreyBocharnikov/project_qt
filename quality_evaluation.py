import copy
import time

import tqdm
from torch.utils.data import DataLoader
from torchvision.models import VGG11_Weights
import torchvision
import torch
from torch.ao.quantization import get_default_qconfig_mapping, default_reuse_input_qconfig, QConfig, HistogramObserver, \
    MinMaxObserver, PerChannelMinMaxObserver, QConfigMapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from MyDataset import ImageNetDataset, IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH

default_per_channel_weight_observer = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
)

default_weight_observer = MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
)


def evaluate(model, transforms, n_samples=1000, batch_size=8, pre_load_dataset=False, return_time=False):
    dataset = ImageNetDataset(IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH, transforms,
                              pre_load=pre_load_dataset, dataset_size=n_samples * batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    match = 0
    times = []
    for i, (tensor, class_id) in enumerate(tqdm.tqdm(dataloader)):
        start = time.time()
        pred = model(tensor)
        times.append(time.time() - start)
        pred_classes = torch.max(pred, dim=1)[1]
        match += (class_id == pred_classes).sum()
        if i == n_samples:
            break
    if return_time:
        return (match / (n_samples * batch_size)), times
    print(match / (n_samples * batch_size))


def quantize(model, qname='x86', do_calibration=True, use_default_mapping=True, per_channel=True, transforms=None):
    x = torch.randn((1, 3, 224, 224), dtype=torch.float)

    if use_default_mapping:
        qconfig_mapping = get_default_qconfig_mapping(qname)
    else:
        if per_channel:
            weight = default_per_channel_weight_observer
        else:
            weight = default_weight_observer

        qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                          weight=weight)

        qconfig_mapping = QConfigMapping() \
                          .set_global(qconfig) \
                          .set_object_type("reshape", default_reuse_input_qconfig)

    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=x)

    if do_calibration:
        print("Calibrating")
        with torch.inference_mode():
            dataset = ImageNetDataset(IMAGE_FOLDER, CSV_TARGETS, CLASS_MAPPING_PATH, transforms)
            for i in tqdm.tqdm(range(1, 501)):
                image_tensor, _ = dataset[-i]
                prepared_model(image_tensor.unsqueeze(dim=0))

    quantized_model = convert_fx(prepared_model)

    return quantized_model


if __name__ == "__main__":
    qengine = 'x86'
    torch.backends.quantized.engine = qengine

    model_fp32 = torchvision.models.vgg11(weights='IMAGENET1K_V1').eval()
    transforms = VGG11_Weights.IMAGENET1K_V1.transforms()

    evaluate(model_fp32, transforms, 100)

    quantized_model_no_calibration = quantize(copy.deepcopy(model_fp32),
                                              do_calibration=False,
                                              transforms=transforms)
    evaluate(quantized_model_no_calibration, transforms, 100)

    quantized_model_with_calibration = quantize(copy.deepcopy(model_fp32),
                                                do_calibration=True,
                                                per_channel=False,
                                                transforms=transforms)
    evaluate(quantized_model_with_calibration, transforms, 100)

    quantized_model_with_calibration = quantize(copy.deepcopy(model_fp32),
                                                do_calibration=True,
                                                per_channel=True,
                                                transforms=transforms)
    evaluate(quantized_model_with_calibration, transforms, 100)

