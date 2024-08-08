from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models import *

import torch_pruning as tp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ofa_specialized_get = torch.hub.load("mit-han-lab/once-for-all", "ofa_specialized_get", verbose=False)


def load_model(name: str) -> nn.Module:
    """
    Load pretrained model from torchvision.

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    model : nn.Module
        Pretrained model.
    """
    if name == "alexnet":
        return alexnet(weights=AlexNet_Weights.DEFAULT)
    elif name == "resnet18":
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif name == "vgg11_bn":
        return vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    elif name == "vgg16_bn":
        return vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)       
    elif name == "vit_b_16":
        return vit_b_16(weights=ViT_B_16_Weights.DEFAULT) 
    elif name == "mobilenet_v3":
        return mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    elif name == "mobilenet_v3_small":
        return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)


def replace_classifier(name: str, model: nn.Module, num_classes: int) -> None:
    """
    Replace the classification layer to output a certain number of classes.

    Parameters
    ----------
    name : str
        The name of the model.
    model : nn.Module
        The model.
    num_classes : The number of classes.
    """
    if name == "resnet18":
        model.fc = nn.Linear(512, num_classes)
    elif name == "resnet50":
        model.fc = nn.Linear(2048, num_classes)
    elif name == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(1024, num_classes)


def get_classifier(name: str, model: nn.Module) -> nn.Module:
    """
    Get the classification layer of a model.

    Parameters
    ----------
    name : str
        The name of the model.
    model : nn.Module
        The model.

    Returns
    -------
    classifier : nn.Module
        The classifier layer.
    """
    if name == "resnet18":
        return model.fc
    elif name == "resnet50":
        return model.fc
        

def load_model_from_pretrained(
    name:             str, 
    path:             str, 
    num_classes:      int, 
    full_load:        bool = False, 
    from_safety:      bool = False,
    from_distributed: bool = False,
) -> nn.Module:
    """
    Load a torchvision vision model from a state dict checkpoint. The final layer output will be replaced 
    with the specified number of classes.

    Parameters
    ----------
    name : str
        Name of model.
    path : str
        Path to state dict checkpoint.
    num_classes : int
        Number of classes the final layer should output.
    full_load : bool
        Load the model from a full save.
    from_safety : bool
        If the model was saved using the safety system.
    from_distributed : bool
        If the model was save with torch.nn.parallel.DistributedDataParallel.

    Returns
    -------
    model : nn.Module
        The pretrained model.
    """
    if full_load:
        return torch.load(path)

    model = None
    if name == "alexnet":
        model = alexnet()
        model.classifier[4] = nn.Linear(4096, 512)
        model.classifier[6] = nn.Linear(512, num_classes)

    elif name == "resnet18":
        model = resnet18()
        model.fc = nn.Linear(512, num_classes)

    elif name == "resnet50":
        model = resnet50()
        model.fc = nn.Linear(2048, num_classes)
        
    elif name == "vgg11_bn":
        model = vgg11_bn()
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "vgg16_bn":
        model = vgg16_bn()
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "vgg16_bn_custom":
        model = vgg16_bn()
        model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        model.classifier[0] = nn.Linear(512 * 4 * 4, 4096)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "vit_b_16":
        model = vit_b_16()
        model.heads.head = nn.Linear(768, num_classes)
    
    elif name == "mobilenet_v3":
        model = mobilenet_v3_large()
        model.classifier[3] = nn.Linear(1280, num_classes)

    elif name == "mobilenet_v3_small":
        model = mobilenet_v3_small()
        model.classifier[3] = nn.Linear(1024, num_classes)

    elif name == "ofa_595M":
        model, _ = ofa_specialized_get("flops@595M_top1@80.0_finetune@75", pretrained=True)
        model.classifier.linear = nn.Linear(1536, num_classes)

    elif name == "ofa_pixel1_20":
        model, _ = ofa_specialized_get("pixel1_lat@20ms_top1@71.4_finetune@25", pretrained=True)
        model.classifier.linear = nn.Linear(1280, num_classes)

    checkpoint = torch.load(path, weights_only=True)
    if from_distributed:
        checkpoint = _remove_module_prefix(checkpoint)

    if from_safety:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    
    return model


def load_vgg_from_pruned(
    path:          str, 
    pruning_ratio: float, 
    dummy_input:   torch.Tensor,
    num_classes:   int = 5,
) -> nn.Module:
    """
    Load a pretrained torchvision VGG16_BN that has been pruned at the layer level.

    Parameters
    ----------
    path : str
        Path to the state dict checkpoint.
    pruning_ratio : float
        Pruning ratio that the model was pruned at.
    dummy_input : torch.Tensor
        Dummy input for tracing.
    num_classes : int
        Number of classes the output layer should have.

    Returns
    -------
    model : nn.Module
        Pretrained pruned VGG.
    """
    model = vgg16_bn()
    model.classifier[6] = nn.Linear(4096, num_classes)
    ignored_layers = [model.classifier[6]]

    model.to("cpu")
    model.eval()

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model             = model,
        example_inputs    = dummy_input,
        importance        = imp,
        pruning_ratio     = pruning_ratio,
        ignored_layers    = ignored_layers,
    )
    pruner.step()

    model.load_state_dict(torch.load(path))
    return model


def load_vgg_custom_from_pruned(
    path:          str, 
    pruning_ratio: float, 
    dummy_input:   torch.Tensor,
    num_classes:   int = 5,
) -> nn.Module:
    """
    Load a pretrained torchvision VGG16_BN that has been pruned at the layer level.

    Parameters
    ----------
    path : str
        Path to the state dict checkpoint.
    pruning_ratio : float
        Pruning ratio that the model was pruned at.
    dummy_input : torch.Tensor
        Dummy input for tracing.
    num_classes : int
        Number of classes the output layer should have.

    Returns
    -------
    model : nn.Module
        Pretrained pruned VGG.
    """
    model = vgg16_bn()
    model.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    model.classifier[0] = nn.Linear(512 * 4 * 4, 4096)
    model.classifier[6] = nn.Linear(4096, num_classes)
    ignored_layers = [model.classifier[6]]

    model.to("cpu")
    model.eval()

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model             = model,
        example_inputs    = dummy_input,
        importance        = imp,
        pruning_ratio     = pruning_ratio,
        ignored_layers    = ignored_layers,
    )
    pruner.step()

    model.load_state_dict(torch.load(path))
    return model


def load_from_layer_pruned(
    model_name:    str,
    path:          str, 
    pruning_ratio: float, 
    dummy_input:   torch.Tensor,
    num_classes:   int = 5,
) -> nn.Module:
    """
    Load a pretrained model that was layer pruned.

    Parameters
    ----------
    model_name : str
        The model name.
    path : str
        Path to the state dict checkpoint.
    pruning_ratio : float
        Pruning ratio that the model was pruned at.
    dummy_input : torch.Tensor
        Dummy input for tracing.
    num_classes : int
        Number of classes the output layer should have.

    Returns
    -------
    model : nn.Module
        Pretrained pruned model.
    """
    model = load_model(model_name)
    replace_classifier(model_name, model, num_classes)
    classifier = get_classifier(model_name, model)

    model.to("cpu")
    model.eval()

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model             = model,
        example_inputs    = dummy_input,
        importance        = imp,
        pruning_ratio     = pruning_ratio,
        ignored_layers    = [classifier],
    )
    pruner.step()

    model.load_state_dict(torch.load(path))
    return model


def _remove_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    return new_state_dict