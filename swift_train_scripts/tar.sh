tar -czf ../verl_no_ckpt.tar.gz \
  --exclude='.git' --exclude='.git/*' \
  --exclude='checkpoints' --exclude='checkpoints/*' \
  --exclude='dialogue_entropy_plots' --exclude='dialogue_entropy_plots/*' \
  --exclude='huggingface.co' --exclude='huggingface.co/*' \
  --exclude='PATH_TO_YOUR_SCORE_ROOT' --exclude='PATH_TO_YOUR_SCORE_ROOT/*' \
  --exclude='outputs' --exclude='outputs/*' \
  --exclude='tensorboard_log' --exclude='tensorboard_log/*' \
  --exclude='*.log' --exclude='*.py' --exclude='*.pdf' --exclude='*.png' \
  --checkpoint=. --totals \
  .
