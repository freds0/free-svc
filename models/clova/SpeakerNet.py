import torch
import torch.nn as nn
import importlib


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, **kwargs):
        super(SpeakerNet, self).__init__()

        if type(model) == str:
            SpeakerNetModel = importlib.import_module(".models." + model).__getattribute__("MainModel")
        else:
            SpeakerNetModel = model
        self.model = SpeakerNetModel(**kwargs)

    def forward(self, data, label=None):

        data = data.reshape(-1, data.size()[-1]).cuda()
        outp = self.model.forward(data)
        return outp

    def loadParameters(self, path):
        print("Loading pretrained model from %s" % (path))
        pretrained_dict = torch.load(path)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print("Pretrained model is loaded.")
