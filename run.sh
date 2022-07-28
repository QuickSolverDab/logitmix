python main.py \
    --network resnet50 \
    --dataset_dir [put your dataset directory] \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.1 \
    --wd 5e-4 \
    --gpu [put your gpu ID for training/testing] \
    --mixmethod ['org', 'mixup', 'cutmix', 'logitmix_M', 'logitmix_C']
