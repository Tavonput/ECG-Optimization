import torch
import torch.nn as nn

from torchvision.models import *

import torch_pruning as tp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ofa_specialized_get = torch.hub.load("mit-han-lab/once-for-all", "ofa_specialized_get")

def load_model(name: str) -> nn.Module:
    match name:
        case "alexnet":
            return alexnet(weights=AlexNet_Weights.DEFAULT)
        case "resnet18":
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        case "resnet50":
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        case "vgg11_bn":
            return vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        case "vgg16_bn":
            return vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)       
        case "vit_b_16":
            return vit_b_16(weights=ViT_B_16_Weights.DEFAULT) 
        case "mobilenet_v3":
            return mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        case "mobilenet_v3_small":
            return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    
def load_model_from_pretrained(name: str, path: str, num_classes: int) -> nn.Module:
    model = None
    match name:
        case "alexnet":
            model = alexnet()
            model.classifier[4] = nn.Linear(4096, 512)
            model.classifier[6] = nn.Linear(512, num_classes)

        case "resnet18":
            model = resnet18()
            model.fc = nn.Linear(512, num_classes)

        case "resnet50":
            model = resnet50()
            model.fc = nn.Linear(2048, num_classes)
            
        case "vgg11_bn":
            model = vgg11_bn()
            model.classifier[6] = nn.Linear(4096, num_classes)

        case "vgg16_bn":
            model = vgg16_bn()
            model.classifier[6] = nn.Linear(4096, num_classes)

        case "vit_b_16":
            model = vit_b_16()
            model.heads.head = nn.Linear(768, num_classes)
        
        case "mobilenet_v3":
            model = mobilenet_v3_large()
            model.classifier[3] = nn.Linear(1280, num_classes)

        case "mobilenet_v3_small":
            model = mobilenet_v3_small()
            model.classifier[3] = nn.Linear(1024, num_classes)

        case "ofa_595M":
            model, _ = ofa_specialized_get("flops@595M_top1@80.0_finetune@75", pretrained=True)
            model.classifier.linear = nn.Linear(1536, num_classes)

        case "ofa_pixel1_20":
            model, _ = ofa_specialized_get("pixel1_lat@20ms_top1@71.4_finetune@25", pretrained=True)
            model.classifier.linear = nn.Linear(1280, num_classes)

    model.load_state_dict(torch.load(path))
    return model

def load_vgg_from_pruned(
    path:          str, 
    pruning_ratio: float, 
    dummy_input:   torch.Tensor,
    num_classes:   int = 5,
) -> nn.Module:
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