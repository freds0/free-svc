import torch
def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor == "pm":
        from models.f0_predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)

    elif f0_predictor == "crepe":
        from models.f0_predictor.CrepeF0Predictor import CrepeF0Predictor
        f0_predictor_object = CrepeF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate, device=kargs["device"],threshold=kargs["threshold"])

    elif f0_predictor == "harvest":
        from models.f0_predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate)

    elif f0_predictor == "dio":
        from models.f0_predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate) 

    elif f0_predictor == "rmvpe":
        from models.f0_predictor.RMVPEF0Predictor import RMVPEF0Predictor
        f0_predictor_object = RMVPEF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate, dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])

    elif f0_predictor == "fcpe":
        from models.f0_predictor.FCPEF0Predictor import FCPEF0Predictor
        f0_predictor_object = FCPEF0Predictor(hop_length=hop_length, sampling_rate=sampling_rate, dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])

    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object
