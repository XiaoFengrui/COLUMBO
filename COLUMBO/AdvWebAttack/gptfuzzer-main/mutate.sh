python Inference/mutate.py \
--lm_name=Inference/models/fine_tune_model_sql_modsecurity \
--ref_lm_name=Inference/models/pretrain_model_sql \
--mutate_times=10 \
--txt_in_len=10 \
--txt_out_len=75 \
--savePath=result/mutated_token_HPD.csv \
--dataPath="/home/ustc-5/XiaoF/AdvWebDefen/Dataset/HPD/test.tsv"