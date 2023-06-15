



#Z
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.00001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05


#caltech
#g_kan
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.0000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2

#gtsrb
#dtl
python src_torch/main.py --phase=moth --suffix=dtl_adv --lr=0.00000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=128 --epochs=2
python src_torch/main.py --phase=moth --suffix=dtl_adv --lr=0.000000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=128 --epochs=2


#dkl
python src_torch/main.py --phase=moth --suffix=dkl_adv --lr=0.00000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=128 --epochs=2
python src_torch/main.py --phase=moth --suffix=dkl_adv --lr=0.000000001 --dataset=gtsrb --model=vgg11_bn --type=adv --batch_size=128 --epochs=2

python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
#brain
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2


#asl
#A

python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.00001 --dataset=asl --model=MobileNet --type=adv --batch_size=32 --epochs=2 --data_ratio=0.05



