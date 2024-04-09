export PYTHONPATH="${PYTHONPATH}:/home/hlpark/merlot_reserve"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

data_dir="/home/hlpark/shared/TVQA"
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'test_dyn' -original_approach 'n' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/test_dyn_new'
python ./finetune/tvqa/prep_data.py -fps 1 -data_dir $data_dir -split 'val_dyn' -original_approach 'n' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/val_dyn_new'