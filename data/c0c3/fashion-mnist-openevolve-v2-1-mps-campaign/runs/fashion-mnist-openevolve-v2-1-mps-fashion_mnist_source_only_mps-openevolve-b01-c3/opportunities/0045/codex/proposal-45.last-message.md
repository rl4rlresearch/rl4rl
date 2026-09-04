MECHANISM: Tail-average-aligned BatchNorm statistics

HYPOTHESIS: Setting BatchNorm momentum to 0.04 will exceed 9,285 correct predictions by aligning running statistics with the effective temporal center of the successful final-10% linear parameter average.

INTENDED_EDIT: Set every BatchNorm2d layer’s running-stat momentum to 0.04 without changing training gradients, parameter averaging, or per-step computation.

EVIDENCE: Linear final-10% weighting improved uniform averaging from 9,282 to 9,285 correct, suggesting better temporal alignment with terminal BatchNorm statistics; momentum 0.04 shifts their effective center toward the linearly averaged parameters while preserving the beneficial full window.

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return ImageClassifier()
=======
def build_model() -> nn.Module:
    model = ImageClassifier()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.04
    return model
>>>>>>> REPLACE