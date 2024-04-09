export PYTHONPATH="${PYTHONPATH}:/home/hlpark/merlot_reserve"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

data_dir="/home/hlpark/shared/TVQA"
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'test' -original_approach 'n' -audio_cut 'n' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/test_new_audalignsub'
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'val' -original_approach 'n' -audio_cut 'n' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/val_new_audalignsub'
