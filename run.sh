#asl
#A
#python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000002 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000003 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#Z
#python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000002 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000003 --dataset=asl --model=MobileNet --type=adv --batch_size=64 --epochs=2 --data_ratio=0.05
#A
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000002 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=A_adv --lr=0.0000003 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05
#Z
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000001 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000002 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05
python src_torch/main.py --phase=moth --suffix=Z_adv --lr=0.0000003 --dataset=asl --model=MobileNet --type=nat --batch_size=64 --epochs=2 --data_ratio=0.05

#caltech
#brain
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.0000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.0000002 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=brain_adv --lr=0.0000003 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
#g_kan
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.0000001 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.0000002 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2
python src_torch/main.py --phase=moth --suffix=g_kan_adv --lr=0.0000003 --dataset=caltech --model=shufflenetv2 --type=adv --batch_size=64 --epochs=2


python src_torch/main.py --phase=moth --suffix=sbg_adv --lr=0.0000003 --dataset=cifar10 --model=resnet50 --type=adv  --batch_size=64 --epochs=2