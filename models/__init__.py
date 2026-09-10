from .baselines import ENSOCNN, ENSOConvLSTM, ENSOGeoformer
from .AtmosFormer import AtmosFormer


MODEL_REGISTRY = {
    "AtmosFormer": AtmosFormer,
    "AtmosFormerSPB": AtmosFormer,
    "ENSOCNN": ENSOCNN,
    "CNN": ENSOCNN,
    "ENSOConvLSTM": ENSOConvLSTM,
    "ConvLSTM": ENSOConvLSTM,
    "ENSOGeoformer": ENSOGeoformer,
    "Geoformer": ENSOGeoformer,
}


def get_model_dict():
    return dict(MODEL_REGISTRY)
