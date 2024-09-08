python main_probs_iff.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs_iff.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs_iff.py --model resnet_binary --save resnet_cifar100_ProbBop2ndOrder_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs_iff.py --model resnet_binary --save resnet_cifar100_ProbBop2ndOrder_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.5, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_binary_iff.py --model resnet_binary --save resnet_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3

python main_binary_iff.py --model resnet_binary --save resnet_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3

python main_binary_iff.py --model resnet_binary --save resnet_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3

python main_binary_iff.py --model resnet_binary --save resnet_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3

