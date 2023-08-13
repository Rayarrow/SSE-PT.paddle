python run.py \
  --dataset_path=data/preprocessed/ml-1m.txt \
  --hidden_units=50 \
  --num_blocks=2 \
  --num_heads=1 \
  --device=0 \
  --test=1 \
  --model_path=res_sse_pt/SSE_PT_epoch_420.pth.tar
