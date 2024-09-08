python main_probs.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_1 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3

python main_probs1.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_2 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3

python main_probs2.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_3 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3

python main_probs3.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_4 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3

python main_probs4.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_5 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3

python main_probs5.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14_6 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3