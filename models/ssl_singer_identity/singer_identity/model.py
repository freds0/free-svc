import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Callable, Union
import torchaudio.transforms as T
from nnAudio import features
import warnings

from .utils.fetch_pretrained import from_hparams, from_scripted
from .models.network_components import (
    get_vision_backbone,
    LogScale,
    Grey2Rgb,
)

HF_SOURCE = "BernardoTorres/singer-identity"


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        spec_layer: str = "melspectogram",
        n_fft: int = 2048,
        hop_length: int = 512,
        **kwargs,
    ):
        super().__init__()

        if spec_layer == "melspectogram":
            n_mels = 128
            if kwargs.get("n_mels", 0):
                n_mels = kwargs["n_mels"]
            self.spec_layer = features.MelSpectrogram(
                n_fft=n_fft, hop_length=hop_length, verbose=False, n_mels=n_mels
            )
        elif spec_layer == "stft":
            self.spec_layer = features.STFT(
                n_fft=n_fft,
                hop_length=hop_length,
                verbose=False,
                output_format="Magnitude" ** kwargs,
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.spec_layer(x)


class Encoder(nn.Module):
    """Encoder, used to extract embeddings from the input acoustic features"""

    def __init__(
        self, backbone="efficientnet_b0", embedding_dim=1000, pretrained=False, **kwargs
    ):
        super().__init__()

        # With attention pooling, not used in the paper
        if backbone == "efficientnet_b0_att":
            encoder_backbone = Efficientnet_att(
                vismod="efficientnet_b0",
                num_classes=1000,
                pretrained=pretrained,
                embedding_dim=embedding_dim,
                **kwargs,
            )
            self.net = nn.Sequential(LogScale(), Grey2Rgb(), encoder_backbone)

        # Default to efficientnet backbone with average pooling, used in the paper
        else:
            encoder_backbone = get_vision_backbone(
                vismod=backbone,
                num_classes=embedding_dim,
                pretrained=pretrained,
                **kwargs,
            )

            # Grey2Rgb() is used to replicate mel-spec channel (efficientnet expects 3)
            self.net = nn.Sequential(LogScale(), Grey2Rgb(), encoder_backbone)

    def forward(self, x):
        """
        x shape [batch, channels, frames]
        """
        embedding = self.net(x)
        return embedding


class Projection(nn.Module):
    """Projection head, used to reduce the dimensionality of the embedding"""
    def __init__(
        self,
        input_dim=1000,
        output_dim=128,
        nonlinearity=None,
        is_identity=False,
        l2_normalize=False,
    ):
        super().__init__()

        self.l2_normalize = l2_normalize
        self.is_identity = is_identity
        self.output_dim = output_dim
        if is_identity:
            self.net = nn.Identity()
        else:
            if nonlinearity is None:
                nonlinearity = torch.nn.SiLU()
            self.net = nn.Sequential(
                nonlinearity, torch.nn.Linear(input_dim, output_dim)
            )

    def forward(self, x):
        projection = self.net(x)
        if self.l2_normalize and not self.is_identity:
            projection = torch.nn.functional.normalize(projection, dim=-1)
        return projection


class IdentityEncoder(nn.Module):
    """Wraps a feature extractor with an encoder, without projection head
    Useful for loading pretrained models"""

    def __init__(self, feature_extractor, encoder):
        super().__init__()
        self.feature_extractor = FeatureExtractor(**feature_extractor)
        self.encoder = Encoder(**encoder)

    def forward(self, x):
        return self.encoder(self.feature_extractor(x))


class SiameseArm(nn.Module):
    """For BYOL"""
    def __init__(
        self,
        encoder: nn.Module,
        projector: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
        normalize_representations: bool = False,
        normalize_projections: bool = False,
    ):
        super().__init__()

        # Encoder
        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.predictor = predictor if predictor is not None else nn.Identity()

        # Normalizations
        self.normalize_y = F.normalize if normalize_representations else nn.Identity()
        self.normalize_z = F.normalize if normalize_projections else nn.Identity()

    def forward(self, x):
        y = self.encoder(x)
        y = self.normalize_y(y)

        z = self.projector(y)
        z = self.normalize_z(z)

        q = self.predictor(z)
        q = self.normalize_z(q)
        return y, z, q


class EncoderWrapper(nn.Module):
    """
    Wraps any encoder with a feature_extractor, encoder and projection parts.
    Projection is set to identity by default.
    Feature extractor can be used to resample signals on-the-fly
        (eg. when a model accepts 16 kHz input).
    """
    def __init__(self, encoder, feature_dim=256, input_sr=44100, output_sr=16000):
        super().__init__()
        self.encoder = encoder
        self.feature_extractor = nn.Sequential(T.Resample(input_sr, output_sr))
        self.encoder = encoder
        self.projection = Projection(input_dim=feature_dim, is_identity=True)
        self.net = nn.Sequential(self.feature_extractor, self.encoder, self.projection)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation: Union[bool, nn.Module] = True,
        use_batchnorm: Union[bool, int] = False,
        batchnorm_fn: Optional[nn.Module] = None,
        last_layer: Optional[nn.Module] = None,
        bias: Optional[bool] = None,
        layer_init: Optional[Union[Callable[[nn.Module], nn.Module], str]] = None,
    ):
        super().__init__()
        self.in_dim = dims[0]
        self.out_dim = dims[-1]

        if len(dims) < 2:
            self.model = nn.Identity()

            if activation or use_batchnorm:
                warnings.warn(
                    "An activation/batch-norm is defined for the projector "
                    "whereas it is the identity function."
                )

        else:
            # define activation layer
            if activation is True:
                activation = nn.ReLU()

            # define batch-norm layer
            if use_batchnorm is not False and batchnorm_fn is None:
                batchnorm_fn = nn.BatchNorm1d

            # useless to add bias just before a batch-norm layer but add the option for completeness
            if bias is None:
                bias = isinstance(use_batchnorm, bool) and not use_batchnorm

            # NOTE: with old implementation use_batchnorm=True means use_batchnorm=False + bias=True
            if use_batchnorm is True:
                use_batchnorm = 0
            elif use_batchnorm is False:
                use_batchnorm = float("inf")

            ckpt_path = None
            if isinstance(layer_init, str):
                ckpt_path = layer_init
                layer_init = lambda x: x
            elif layer_init is None:
                layer_init = lambda x: x

            layers = []

            output_dim = dims.pop()

            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                layers.append(layer_init(nn.Linear(in_dim, out_dim, bias=bias)))
                if i >= use_batchnorm:
                    layers.append(batchnorm_fn(out_dim))
                if activation:
                    layers.append(activation)
            layers.append(nn.Linear(dims.pop(), output_dim, bias=True))
            if last_layer is not None:
                layers.append(last_layer)

            self.model = nn.Sequential(*layers)

            if ckpt_path is not None:
                self.load_state_dict(torch.load(ckpt_path))

    def forward(self, x):
        x = self.model(x)
        return x


# Not used in the experiments in the paper
class Efficientnet_att(nn.Module):
    def __init__(
        self,
        vismod="efficientnet_b0",
        num_classes=1000,
        pretrained=False,
        embedding_dim=1000,
        **kwargs,
    ):
        super(Efficientnet_att, self).__init__()

        self.vision = get_vision_backbone(
            vismod=vismod, num_classes=num_classes, pretrained=pretrained, **kwargs
        ).features

        self.att = nn.Sequential(
            nn.Conv1d(1280, int(embedding_dim / 2), kernel_size=1, groups=2),
            AttentiveStatisticPool(int(embedding_dim / 2), 128),
        )

        self.avg = nn.AvgPool2d((4, 1))
        self.bn1 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        y = self.vision(x)
        y = self.avg(y).squeeze(2)
        y = self.att(y)
        return self.bn1(y)


# Not used in the experiments in the paper
class AttentiveStatisticPool(nn.Module):
    def __init__(self, c_in, c_mid):
        super(AttentiveStatisticPool, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1),
            nn.Tanh(),  # seems like most implementations uses tanh?
            nn.Conv1d(c_mid, c_in, kernel_size=1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # x.shape: B x C x t
        alpha = self.network(x)
        mu_hat = torch.sum(alpha * x, dim=-1)
        var = torch.sum(alpha * x**2, dim=-1) - mu_hat**2
        std_hat = torch.sqrt(var.clamp(min=1e-9))
        y = torch.cat([mu_hat, std_hat], dim=-1)
        # y.shape: B x (c_in*2)
        return y


def load_model(
    model, source=HF_SOURCE, torchscript=False, savedir=None, input_sr=44100
):
    """Load a model from a source, can be a local path or a huggingface model hub ID"""

    if torchscript:
        if input_sr != 44100:
            raise Exception("Torchscript models only support 44100 Hz input")
        model = from_scripted(f"{model}/model.ts", source, savedir=savedir)
    elif "." in model:
        # Instantiate IdentityEncoder with input_sr argument
        model = from_hparams(
            IdentityEncoder,
            source,
            hparams_file=f"{model}/hyperparams.yaml",
            weights_file=f"{model}/model.pt",
            savedir=savedir,
        )
    else:
        # CHeck
        model = from_hparams(
            IdentityEncoder,
            source,
            hparams_file=f"{model}/hyperparams.yaml",
            weights_file=f"{model}/model.pt",
            savedir=savedir,
        )
    if input_sr != 44100:
        # Replace feature extractor with a resampler
        feature_extractor = model.feature_extractor
        model.feature_extractor = nn.Sequential(
            T.Resample(input_sr, 44100), feature_extractor
        )
        print(f"Resampling input from {input_sr} to 44100 Hz")

    return model
