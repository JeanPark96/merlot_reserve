export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

# config="/home/hlpark/merlot_reserve/pretrain/configs/tvqa_finetune_large.yaml"
config="/home/hlpark/merlot_reserve/pretrain/configs/tvqa_finetune_base.yaml"
ckpt="/home/hlpark/.cache/merlotreserve/tvqa_finetune_base"
file_path="gs://merlot_test_tvqa/original_tvqa_record"
model_mn=""
suffix="_orig2"
python ./submit_to_leaderboard.py -pretrain_config_file $config -ckpt $ckpt -file_path $file_path -suffix $suffix