CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH="$(pwd):$(pwd)/packages/DFA3D:$(pwd)/packages/bev_pool:${PYTHONPATH:-}" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/main.py \
      --config_path /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/configs/customs/FoundationSSC-SemanticKITTI.py \
      --log_folder /c20250502/wangyushen/Outputs/foundationssc/sem_kitti/train \
      --log_every_n_steps 50