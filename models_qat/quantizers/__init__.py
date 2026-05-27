from .dorefa import DoReFaWeightQuantizer, DoReFaActQuantizer
# 未来如果你写了 LSQ，在这里 import 进来：
# from .lsq import LSQWeightQuantizer, LSQActQuantizer

def get_weight_quantizer(qat_method, wbits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaWeightQuantizer(wbits)
    # elif qat_method == 'lsq':
    #     return LSQWeightQuantizer(wbits)
    else:
        raise ValueError(f"Unsupported QAT method for weights: {qat_method}")

def get_act_quantizer(qat_method, abits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaActQuantizer(abits)
    # elif qat_method == 'lsq':
    #     return LSQActQuantizer(abits)
    else:
        raise ValueError(f"Unsupported QAT method for activations: {qat_method}")
