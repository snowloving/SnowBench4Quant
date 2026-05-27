from .dorefa import DoReFaWeightQuantizer, DoReFaActQuantizer
from .lsq import LSQWeightQuantizer, LSQActQuantizer
from .pact import PACTWeightQuantizer, PACTActQuantizer # 新增引入

def get_weight_quantizer(qat_method, wbits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaWeightQuantizer(wbits)
    elif qat_method == 'lsq':
        return LSQWeightQuantizer(wbits)
    elif qat_method == 'pact': # 新增分支
        return PACTWeightQuantizer(wbits)
    else:
        raise ValueError(f"Unsupported QAT method for weights: {qat_method}")

def get_act_quantizer(qat_method, abits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaActQuantizer(abits)
    elif qat_method == 'lsq':
        return LSQActQuantizer(abits)
    elif qat_method == 'pact': # 新增分支
        return PACTActQuantizer(abits)
    else:
        raise ValueError(f"Unsupported QAT method for activations: {qat_method}")
