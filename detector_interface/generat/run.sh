python generat_interface.py  --task_name adv-sst-2 --model_config ../../GenerAT-main/deberta-v3-large/rtd_large.json \
                             --init_generator ../../GenerAT-main/deberta-v3-large/pytorch_model.generator.bin \
	                         --init_discriminator ../../GenerAT-main/deberta-v3-large/pytorch_model.bin \
                             --data_dir /home/ustc-5/XiaoF/AdvWebDefen/Dataset/SIK \
                             --output_dir ./output/ \
                             --detector GenerAT \
                             --dataset SIK \
                             --tag deberta-v3-large \
                             