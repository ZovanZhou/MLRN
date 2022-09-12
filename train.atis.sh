CUDA_VISIBLE_DEVICES=0 python main.py \
    --seed 0 \
    --order 1 \
    --lr 1e-5 \
    --epoch 100 \
    --opt adamw \
    --gamma 0.6 \
    --dataset atis \
    --batch_size 16 \
    --dropout_rate 0.4 \
    --graph_attn_heads 4 \
    --graph_output_dim 256 \
    --rm_num \