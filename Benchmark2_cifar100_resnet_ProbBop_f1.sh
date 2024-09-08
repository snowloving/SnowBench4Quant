python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a01_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.1,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a02_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.2,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a03_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.3,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a04_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.4,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a05_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a07_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.7,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a08_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.8,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2

python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_a09_d20_f1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop','gamma':1e-4,'alpha':0.9,'threshold':1e-8, 'decay':20, 'formula':1}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2