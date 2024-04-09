export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

config="/home/hlpark/merlot_reserve/pretrain/configs/tvqa_finetune_val_orig.yaml"
ckpt="/home/hlpark/.cache/merlotreserve/tvqa_finetune_base"
file_path="gs://merlot_test_tvqa/hirest_tvqa_record"
model_mn="_sb"
suffix="_new_audalignsub"

python ./submit_to_leaderboard.py -pretrain_config_file $config -ckpt $ckpt -file_path $file_path -suffix $suffix -model_mn $model_mn

model_mn="_dyn"
python ./submit_to_leaderboard.py -pretrain_config_file $config -ckpt $ckpt -file_path $file_path -suffix $suffix -model_mn $model_mn

file_path="gs://merlot_test_tvqa/original_tvqa_record"
python ./submit_to_leaderboard.py -pretrain_config_file $config -ckpt $ckpt -file_path $file_path -suffix $suffix