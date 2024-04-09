config="/home/hlpark/merlot_reserve/pretrain/configs/tvqa_finetune_base.yaml"
ckpt="/home/hlpark/.cache/merlotreserve/tvqa_finetune_base"

python ./finetune/tvqa/submit_to_leaderboard.py --pretrain_config_file $conifg --ckpt $ckpt 