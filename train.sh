python run.py \
  --dataset_path=data/preprocessed/ml-1m.txt \
  --hidden_units=50 \
  --dropout=0.2 \
  --num_blocks=2 \
  --num_heads=1 \
  --device=0 \
  --val_interval=1000 \
  --sse_prob_user=0.1 \
  --save_folder=res_sse_pt
