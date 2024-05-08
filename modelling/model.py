import torch
import torch.nn as nn
import torchvision.models as models

class CEEffNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=2, model_name='effb0', dropout=0.4):
        super(CEEffNet, self).__init__()

        self.encoder = choose_model(model_name=model_name)
        dim_in = self.encoder.classifier[1].in_features
        self.encoder.classifier[1] = nn.Identity()
        self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dim_in, int(dim_in/2)),
                Swish_Module(),
                nn.Dropout(p=dropout),
                nn.Linear(int(dim_in/2), num_classes))

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)


class CEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=2, model_name='res50', dropout=0.4):
        super(CEResNet, self).__init__()
        self.encoder = choose_model(model_name=model_name)
        dim_in = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()
        self.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dim_in, int(dim_in/2)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(int(dim_in/2), num_classes))

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)

def choose_model(model_name : str) -> nn.Module:
    if 'res' in model_name:
        if '18' in model_name:
            feature_extractor = models.resnet18(pretrained=True)
        elif '34' in model_name:
            feature_extractor = models.resnet34(pretrained=True)
        elif '50' in model_name:
            feature_extractor = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))

    elif 'eff' in model_name:
        if 'b0' in model_name:
            feature_extractor = models.efficientnet_b0(pretrained=True)
        elif 'b1' in model_name:
            feature_extractor = models.efficientnet_b1(pretrained=True)
        elif 'b2' in model_name:
            feature_extractor = models.efficientnet_b2(pretrained=True)
        else:
            raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))
    else:
        raise NotImplementedError("The feature extractor cannot be instantiated: model asked -> {} does not exist".format(model_name))
    
    return feature_extractor

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod 
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
