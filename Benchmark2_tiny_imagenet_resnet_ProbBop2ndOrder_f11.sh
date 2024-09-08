python main_probs.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_1 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2

python main_probs1.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_2 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2

python main_probs2.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_3 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2

python main_probs3.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_4 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2

python main_probs4.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_5 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2

python main_probs5.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11_6 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2