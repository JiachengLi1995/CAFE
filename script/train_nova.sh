DATASET=${2}

CUDA_VISIBLE_DEVICES=$1 python3.7 large_gpu.py \
    --data_path "data/${DATASET}" \
    --train_batch_size 64 \
    --val_batch_size 64 \
    --test_batch_size 64 \
    --test_negative_sampler_code 'popular' \
    --test_negative_sample_size 100 \
    --test_negative_sampling_seed 98765 \
    --device 'cuda' \
    --device_idx $1 \
    --optimizer 'Adam' \
    --lr 1e-3 \
    --weight_decay 0 \
    --num_epochs 1000 \
    --best_metric 'NDCG@10' \
    --model_init_seed 0 \
    --trm_dropout 0.3 \
    --trm_att_dropout 0.3 \
    --trm_hidden_dim 128 \
    --trm_max_len 50 \
    --trm_num_blocks 2  \
    --trm_num_heads 1 \
    --verbose 10 \
    --model_code 'nova'