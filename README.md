训练
python main.py --datadir ./market1501/ --batchid 16 --batchtest 32 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --resume 0 --margin 1.2 --save adam_1 --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --reset --re_rank --amsgrad
测试
python main.py --datadir ./market1501/ --save adam_1 --test_only --re-rank
python search.py --save adam_1
运行
python main_window.py
![示例图片](image.png)
