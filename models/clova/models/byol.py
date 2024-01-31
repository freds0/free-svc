# -*- encoding: utf-8 -*-

from .ssl_singer_identity.singer_identity import load_model


def MainModel(**kwargs):

    model = load_model("byol", torchscript=True)
    model.train()
    return model
