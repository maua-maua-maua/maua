# CogVideo

Based on https://github.com/lkwq007/CogVideo-low-vram
Image prompting from https://github.com/NightmareAI/CogVideo by @neverix

## Usage Example
```bash
python -m maua.autoregressive.cog.video.generate \
  --text "moving from right to left" \
  --image /path/to/image.jpg \
  --output-path output/ \
  --translate \
  --both-stages \
  --use-guidance-stage1 \
  --tokenizer-type fake \
  --mode inference \
  --distributed-backend nccl \
  --fp16 \
  --temperature 1.05 \
  --top_k 12
```