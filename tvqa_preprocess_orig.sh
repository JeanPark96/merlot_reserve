export PYTHONPATH="${PYTHONPATH}:/home/hlpark/merlot_reserve"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

data_dir="/home/hlpark/shared/TVQA"
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'test' -original_approach 'y' -audio_cut 'y' -use_subtitle 'n' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/test_orig_nosub'
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'val' -original_approach 'y' -audio_cut 'y' -use_subtitle 'n' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/val_orig_nosub'
