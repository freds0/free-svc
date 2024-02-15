from s3prl.upstream.wav2vec2.wav2vec2_model import MultiheadAttention
from torch import nn


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def init_module(m: nn.Module):
    for p in m.parameters():
        nn.init.normal_(p, mean=0, std=0.02)


def init_module_bert(m: nn.Module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(m, nn.Linear):
        normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.Embedding):
        normal_(m.weight.data)
        if m.padding_idx is not None:
            m.weight.data[m.padding_idx].zero_()
    if isinstance(m, MultiheadAttention):
        normal_(m.q_proj.weight.data)
        normal_(m.k_proj.weight.data)
        normal_(m.v_proj.weight.data)


def init_module_cnn(m: nn.Module):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.LayerNorm):
        m.reset_parameters()


def init_module_pos_conv(m: nn.Module):
    if isinstance(m, nn.Conv1d):
        m.reset_parameters()
    if isinstance(m, nn.LayerNorm):
        m.reset_parameters()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
