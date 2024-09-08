python main_probs.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14_1 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs1.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14_2 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs2.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14_3 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs3.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14_4 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3

python main_probs4.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14_5 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3