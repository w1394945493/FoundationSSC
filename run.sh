# 一. SemanticKiTTi
# 1.生成 SSC 训练标签
# 把 SemanticKITTI 压缩体素标签转换成三个项目可直接读取的 NumPy 标签。
# semantickitti: SSC 体素样本：共 8,550 帧 有真值的 SSC 样本（序列 00–10）：4,649 帧 无公开真值的测试样本（序列 11–21）：3,901 帧
# 训练集 00–07、09、10：3834
# 验证集 08：            815
# 测试集 11–21：        3901
# 总计：                8550
# 统计原始体素标签
find /c20250502/wangyushen/Datasets/kitti/semantickitti/dataset/sequences \
  -path '*/voxels/*.label' -type f | wc -l
# 统计无效体素掩码
find /c20250502/wangyushen/Datasets/kitti/semantickitti/dataset/sequences \
  -path '*/voxels/*.invalid' -type f | wc -l
# 统计已经生成的 NumPy 标签
find /c20250502/wangyushen/Datasets/kitti/semantickitti/dataset/labels \
  -type f -name '*_1_1.npy' | wc -l

cd /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/tools
conda activate /vepfs-mlp2/c20250502/haoce/wangyushen/conda_env/wangyushentemp
python /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/tools/preprocess.py \
  --kitti_root /c20250502/wangyushen/Datasets/kitti/semantickitti \
  --kitti_preprocess_root /c20250502/wangyushen/Datasets/kitti/semantickitti/dataset

# 2.需要 lidarseg：不需要复制数据，建立软链接：
SEM_DATA=/c20250502/wangyushen/Datasets/kitti/semantickitti/dataset

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  mkdir -p "$SEM_DATA/lidarseg/$seq"
  ln -sfn "../../sequences/$seq/labels" \
    "$SEM_DATA/lidarseg/$seq/labels"
done
# 统计通过软链接访问的逐点标签：预期23201个
find -L "$SEM_DATA/lidarseg" \
  -type f \
  -name '*.label' | wc -l

# FoundationSSC 各项用途
# 内容	是否必需	用途
# image_2/*.png	必需	左目图像
# image_3/*.png	必需	右目图像，供 FoundationStereo 使用
# calib.txt	必需	相机和 LiDAR 标定
# velodyne/*.bin	必需	在线生成稀疏深度监督
# voxels/*.bin	必需	确定标准 SSC 样本帧
# labels/*_1_1.npy	必需	三维 SSC 真值
# lidarseg/*/labels/*.label	默认完整模型需要	生成稀疏二维语义监督
# depth/*.npy	不需要	FoundationSSC 不使用 MobileStereoNet 预生成深度
# poses.txt	默认训练不需要	当前加载器未读取
# times.txt	默认训练不需要	当前加载器未读取
# voxels/*.occluded	预处理后不直接读取	不需要额外处理
# *_1_2.npy	默认配置不直接读取	预处理脚本会顺便生成

# 二. SSCBench-Kitti-360

#--------------------------------------------------#
cd /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/packages/DFA3D
python3 setup.py build_ext --inplace
cd /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/packages/bev_pool
python3 setup.py build_ext --inplace
# 修改/vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/mmdet3d_plugin/models/backbones/FoundationStereo/core/extractor.py预训练权重路径

#--------------------------------------------------#
# 评估
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd):$(pwd)/packages/DFA3D:$(pwd)/packages/bev_pool:${PYTHONPATH:-}" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/main.py \
      --eval \
      --ckpt_path /c20250502/wangyushen/Weights/foundationssc/FoundationSSC-KITTI360.ckpt \
      --config_path /vepfs-mlp2/c20250502/haoce/wangyushen/FoundationSSC/configs/customs/FoundationSSC-KITTI360.py \
      --log_folder /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/foundationssc/kitti360/val \
      --log_every_n_steps 50
