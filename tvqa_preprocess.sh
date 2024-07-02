export PYTHONPATH="${PYTHONPATH}:/home/hlpark/merlot_reserve"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hlpark/gcloud_credential_mreserve.json"

data_dir="/home/hlpark/shared/TVQA"

# python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'test' -original_approach 'y' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/test_orig2'
# python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'val' -original_approach 'y' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/val_orig2'

python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'test' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/test_new_audiofixed'
python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'val' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/original_tvqa_record/val_new_audiofixed'

python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'test_sb' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/test_sb_new_audiofixed'
python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'val_sb' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/val_sb_new_audiofixed'

python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'test_dyn' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/test_dyn_new_audiofixed'
python ./finetune/tvqa/prep_data.py -fps 3 -data_dir $data_dir -split 'val_dyn' -original_approach 'n' -audio_cut 'f' -base_fn 'gs://merlot_test_tvqa/hirest_tvqa_record/val_dyn_new_audiofixed'