python Inference/inference.py \
--lm_name=Inference/models/fine_tune_model_sql_modsecurity \
--ref_lm_name=Inference/models/pretrain_model_sql \
--total_nums=10 \
--txt_in_len=10 \
--txt_out_len=128 \
--savePath=result/sql_mod.csv \
--dataPath="/home/ustc-5/XiaoF/AdvWebDefen/Dataset/SIK/test.tsv"