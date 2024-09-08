python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11_1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0

python main_probs1.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11_2 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0

python main_probs2.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11_3 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0

python main_probs3.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11_4 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0

python main_probs4.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11_5 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
