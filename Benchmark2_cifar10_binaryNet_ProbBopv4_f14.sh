python main_probs.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14_1 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1

python main_probs1.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14_2 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1

python main_probs2.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14_3 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1

python main_probs3.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14_4 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1

python main_probs4.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14_5 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1