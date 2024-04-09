export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

config="/home/hlpark/merlot_reserve/pretrain/configs/tvqa_finetune_val_orig.yaml"
ckpt="/home/hlpark/.cache/merlotreserve/tvqa_finetune_base"
file_path="gs://merlot_test_tvqa/hirest_tvqa_record"
model_mn="_dyn"
suffix="_new"
python ./submit_to_leaderboard.py -pretrain_config_file $config -ckpt $ckpt -file_path $file_path -suffix $suffix -model_mn $model_mn