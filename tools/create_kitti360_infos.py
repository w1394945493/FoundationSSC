#!/usr/bin/env python3
"""为 FoundationSSC 的 KITTI-360 数据集生成 train/val/test 元信息文件。"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


# 与 KITTI360Dataset 中保持一致的数据划分：每个划分对应若干行车序列。
SPLITS = {
    "train": [
        "2013_05_28_drive_0000_sync",
        "2013_05_28_drive_0002_sync",
        "2013_05_28_drive_0003_sync",
        "2013_05_28_drive_0004_sync",
        "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0007_sync",
        "2013_05_28_drive_0010_sync",
    ],
    "val": ["2013_05_28_drive_0006_sync"],
    "test": ["2013_05_28_drive_0009_sync"],
}


def parse_args():
    """解析数据路径、输出路径以及生成方式等命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成 train/val/test 三份 KITTI-360 metadata pkl"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="KITTI-360 数据集根目录",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        required=True,
        help="占用标签根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="pkl 输出目录；默认使用 data_root/preprocess",
    )
    parser.add_argument(
        "--load-continuous",
        action="store_true",
        help="以左相机全部 PNG 为索引；默认与当前 Dataset 一样以 voxels/*.bin 为索引",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 pkl 文件",
    )
    return parser.parse_args()


def read_calib():
    """返回与 KITTI360Dataset.read_calib() 相同的固定标定矩阵。"""
    # P2 和 P3：分别将左、右相机坐标系中的三维点投影到二维图像平面。
    p2 = np.array(
        [
            [552.554261, 0.0, 682.049453, 0.0],
            [0.0, 552.554261, 238.769549, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    p3 = np.array(
        [
            [552.554261, 0.0, 682.049453, -328.318735],
            [0.0, 552.554261, 238.769549, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # 相机坐标系到激光雷达坐标系的刚体变换矩阵。
    cam2velo = np.array(
        [
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # 数据 pipeline 使用的是激光雷达到相机的变换，因此对 cam2velo 求逆。
    velo2cam = np.linalg.inv(cam2velo)

    # 将原始 3×4 相机投影矩阵扩展成齐次形式的 4×4 矩阵，方便矩阵连乘。
    p2_h = np.eye(4)
    p3_h = np.eye(4)
    p2_h[:3, :4] = p2
    p3_h[:3, :4] = p3
    return p2_h, p3_h, velo2cam


def create_infos(data_root, ann_file, sequences, load_continuous=False):
    """按 KITTI360Dataset.load_annotations() 的格式创建一个划分的样本列表。

    这里只扫描文件并记录路径、帧编号和标定参数，不会读取图像、点云或
    占用标签的实际内容。真正的数据内容仍由 Dataset.__getitem__ 按需加载。
    """
    # 标定参数在同一个数据集划分的所有帧之间共用，只需计算一次。
    p2, p3, velo2cam = read_calib()

    # 激光雷达到图像平面的完整投影：先 velo2cam，再经过相机投影矩阵。
    proj_matrix_2 = p2 @ velo2cam
    proj_matrix_3 = p3 @ velo2cam

    # infos 中每个字典对应一帧，最终会整体序列化到 pkl 文件。
    infos = []

    # 逐个扫描当前 train/val/test 划分包含的行车序列。
    for sequence in sequences:
        image_base = data_root / "data_2d_raw" / sequence
        if load_continuous:
            # 连续模式使用左相机目录中的全部图片作为待处理帧。
            index_pattern = image_base / "image_00/data_rect/*.png"
        else:
            # 默认模式与原 Dataset 一致，通过 voxels/*.bin 确定有效帧编号。
            index_pattern = image_base / "voxels/*.bin"

        # 展开通配符并排序，使每次生成的样本顺序保持一致。
        frame_paths = sorted(glob.glob(str(index_pattern)))

        # tqdm 展示当前序列的完成比例、速度和预计剩余时间。
        for frame_path in tqdm(
            frame_paths,
            desc=sequence,
            unit="frame",
            dynamic_ncols=True,
        ):
            # 例如从 0000000123.bin 中提取帧编号 0000000123。
            frame_id = Path(frame_path).stem

            # SSCBench 的占用标签命名规则为 <frame_id>_1_1.npy。
            voxel_path = ann_file / sequence / f"{frame_id}_1_1.npy"

            # 保存后续 Dataset.get_data_info() 和数据 pipeline 所需的全部元信息。
            infos.append(
                {
                    # image_00 和 image_01 分别对应左、右相机。
                    "img_2_path": str(
                        image_base / "image_00" / "data_rect" / f"{frame_id}.png"
                    ),
                    "img_3_path": str(
                        image_base / "image_01" / "data_rect" / f"{frame_id}.png"
                    ),
                    "sequence": sequence,
                    "frame_id": frame_id,
                    # 相机投影、坐标变换以及激光雷达到图像的组合投影矩阵。
                    "P2": p2,
                    "P3": p3,
                    "T_velo_2_cam": velo2cam,
                    "proj_matrix_2": proj_matrix_2,
                    "proj_matrix_3": proj_matrix_3,
                    # 官方测试帧可能没有标签；不存在时记录 None，由 pipeline
                    # 创建占位标签，避免后续尝试打开不存在的文件。
                    "voxel_path": str(voxel_path) if voxel_path.exists() else None,
                }
            )

    return infos


def dump_pickle(infos, output_path, overwrite=False):
    """将样本元信息安全写入 pkl，并避免意外覆盖已有文件。"""
    # 默认保护已经生成的文件；显式传入 --overwrite 后才允许覆盖。
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} 已存在；如需重新生成，请添加 --overwrite"
        )

    # 自动创建输出目录。先写临时文件，完整写入后再原子替换目标文件，
    # 避免程序中断时留下看似存在但内容不完整的 pkl。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(infos, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)


def main():
    """依次生成 train、val、test 三个划分的 metadata 文件。"""
    args = parse_args()

    # 展开路径中的 ~ 并转换为绝对路径，保证 pkl 中记录的文件位置明确。
    data_root = args.data_root.expanduser().resolve()
    ann_file = args.ann_file.expanduser().resolve()

    # 未指定 --output-dir 时，将 pkl 放到数据集根目录的 preprocess 下。
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "preprocess"
    )

    print(f"数据根目录：{data_root}")
    print(f"标签根目录：{ann_file}")
    print(f"输出目录：{output_dir}")

    # Python 字典保持插入顺序，因此按 train → val → test 依次生成。
    for split, sequences in SPLITS.items():
        print(f"\n开始生成 {split} metadata")
        infos = create_infos(
            data_root=data_root,
            ann_file=ann_file,
            sequences=sequences,
            load_continuous=args.load_continuous,
        )
        # 每个划分单独保存，Dataset 初始化时只需加载对应的一份文件。
        output_path = output_dir / f"kitti360_infos_{split}.pkl"
        dump_pickle(infos, output_path, overwrite=args.overwrite)
        print(f"已保存 {len(infos)} 条样本：{output_path}")


if __name__ == "__main__":
    main()
