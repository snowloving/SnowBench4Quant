from .dorefa import DoReFaWeightQuantizer, DoReFaActQuantizer
from .lsq import LSQWeightQuantizer, LSQActQuantizer
from .pact import PACTWeightQuantizer, PACTActQuantizer
from .lsq_plus import LSQPlusWeightQuantizer, LSQPlusActQuantizer # 引入 LSQ+

def get_weight_quantizer(qat_method, wbits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaWeightQuantizer(wbits)
    elif qat_method == 'pact':
        return PACTWeightQuantizer(wbits)
    elif qat_method == 'lsq':
        return LSQWeightQuantizer(wbits)
    elif qat_method == 'lsq_plus': # 👈 新增 LSQ+
        return LSQPlusWeightQuantizer(wbits)
    else:
        raise ValueError(f"Unsupported QAT method for weights: {qat_method}")

def get_act_quantizer(qat_method, abits):
    qat_method = qat_method.lower()
    if qat_method == 'dorefa':
        return DoReFaActQuantizer(abits)
    elif qat_method == 'pact':
        return PACTActQuantizer(abits)
    elif qat_method == 'lsq':
        return LSQActQuantizer(abits)
    elif qat_method == 'lsq_plus': # 👈 新增 LSQ+
        return LSQPlusActQuantizer(abits)
    else:
        raise ValueError(f"Unsupported QAT method for activations: {qat_method}")
