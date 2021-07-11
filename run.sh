# python=3.9
# torch=3.9
# transformers=4.8

CUDA_VISIBLE_DEVICES=3 python main.py --task re --bs 32 --lr 5e-5 --epoch 30 --model_name backup --attention --warmup_proportion -1
CUDA_VISIBLE_DEVICES=2 python main.py --lr 1e-5 --tlr 5e-4 --bs 16 --model_name ace05