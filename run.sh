
#cifar resnet18
#green
python src_torch/main.py --phase=moth --suffix=green_adv --lr=0.00000001 --dataset=cifar10 --model=resnet18 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=green_adv --lr=0.000000001 --dataset=cifar10 --model=resnet18 --type=adv --batch_size=64 --epochs=2

#sbg
python src_torch/main.py --phase=moth --suffix=sbg_adv --lr=0.00000001 --dataset=cifar10 --model=resnet18 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=sbg_adv --lr=0.000000001 --dataset=cifar10 --model=resnet18 --type=adv  --batch_size=64 --epochs=2


#cifar resnet50
#green
python src_torch/main.py --phase=moth --suffix=green_adv --lr=0.00000001 --dataset=cifar10 --model=resnet50 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=green_adv --lr=0.000000001 --dataset=cifar10 --model=resnet50 --type=adv --batch_size=64 --epochs=2


#sbg
python src_torch/main.py --phase=moth --suffix=sbg_adv --lr=0.00000001 --dataset=cifar10 --model=resnet50 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=sbg_adv --lr=0.000000001 --dataset=cifar10 --model=resnet50 --type=adv  --batch_size=64 --epochs=2



#gtsrb
#dtl
python src_torch/main.py --phase=moth --suffix=dtl_adv --lr=0.00000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=dtl_adv --lr=0.000000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=64 --epochs=2


#dkl
python src_torch/main.py --phase=moth --suffix=dkl_adv --lr=0.00000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=dkl_adv --lr=0.000000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=64 --epochs=2


#fmnist
#stripet
python src_torch/main.py --phase=moth --suffix=stripet_adv --lr=0.000001 --dataset=fmnist --model=MobileNetV2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=stripet_adv --lr=0.00000001 --dataset=fmnist --model=MobileNetV2 --type=adv --batch_size=64 --epochs=2


#plaids
python src_torch/main.py --phase=moth --suffix=plaids_adv --lr=0.000001 --dataset=fmnist --model=MobileNetV2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=plaids_adv --lr=0.00000001 --dataset=fmnist --model=MobileNetV2 --type=adv --batch_size=64 --epochs=2


#mnistm
#blue 8
python src_torch/main.py --phase=moth --suffix=blue_adv --lr=0.000001 --dataset=mnistm --model=densenet --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=blue_adv --lr=0.00001 --dataset=mnistm --model=densenet --type=adv --batch_size=64 --epochs=2

#black 2
python src_torch/main.py --phase=moth --suffix=black_adv --lr=0.000001 --dataset=mnistm --model=densenet --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=black_adv --lr=0.00001 --dataset=mnistm --model=densenet --type=adv --batch_size=64 --epochs=2


#asl
#A
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.00000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05
#Z
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.00000001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05

#caltech
#brain
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.00000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2

#g_kan
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.00000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2



