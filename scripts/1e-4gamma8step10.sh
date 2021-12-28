python main.py \
--dataset_name mimic_cxr \
--image_dir /GPUFS/nsccgz_ywang_zfd/limengfei/datasets/mimic_cxr/images/ \
--ann_path /GPUFS/nsccgz_ywang_zfd/limengfei/datasets/mimic_cxr/annotation.json \
--max_seq_length 100 \
--threshold 10 \
--epochs 100 \
--gamma 0.8 \
--step_size 10 \
--batch_size 256 \
--seed 456789 \
--img_size 224 \
--n_gpu 4 \
--num_workers 16 \
--lr_res 5e-5 \
--lr_lp 5e-5 \
--lr_ed 1e-4 \
--model_name 1e-4gamma8step10 \
--resume /GPUFS/nsccgz_ywang_zfd/limengfei/ReportGen/results/1e-4gamma8step10/current_checkpoint.pth