#!/usr/bin/env python3
"""生成 FoundationSSC 使用的 SemanticKITTI 样本元信息 pkl。"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


#* 与 SemanticKITTIDataset 保持一致；普通 test 与 val 都是 08，因此共用 val.pkl。
SPLITS = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "val": ["08"],
    "test_submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
}


def parse_args():
    """解析数据根目录、标签目录和输出方式。"""
    parser = argparse.ArgumentParser(
        description="生成 train/val/test_submit 三份 SemanticKITTI metadata pkl"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="SemanticKITTI dataset 根目录，内部应包含 sequences",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        required=True,
        help="占用标签根目录，内部按序列号存放 *_1_1.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="pkl 输出目录；默认使用 data_root/metadata",
    )
    parser.add_argument(
        "--load-continuous",
        action="store_true",
        help="以 image_2/*.png 为索引；默认与 Dataset 一样以 voxels/*.bin 为索引",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 pkl 文件",
    )
    return parser.parse_args()


def read_calib(calib_path):
    """读取单个序列的 calib.txt，返回齐次形式的 P2、P3 和 Tr。"""
    calib_all = {}
    with calib_path.open("r") as file:
        for line in file:
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(item) for item in value.split()])

    # 将原始 3×4 矩阵扩展成 4×4 齐次矩阵，与 Dataset 当前格式一致。
    p2 = np.eye(4)
    p3 = np.eye(4)
    velo2cam = np.eye(4)
    p2[:3, :4] = calib_all["P2"].reshape(3, 4)
    p3[:3, :4] = calib_all["P3"].reshape(3, 4)
    velo2cam[:3, :4] = calib_all["Tr"].reshape(3, 4)
    return p2, p3, velo2cam


def create_infos(data_root, ann_file, sequences, load_continuous=False):
    """扫描一个数据划分并生成与 SemanticKITTIDataset 一致的元信息列表。"""
    infos = []

    #* 每个 SemanticKITTI 序列拥有独立 calib.txt，必须逐序列读取标定矩阵。
    for sequence in sequences:
        sequence_root = data_root / "sequences" / sequence
        calib_path = sequence_root / "calib.txt"
        if not calib_path.is_file():
            raise FileNotFoundError(f"找不到序列 {sequence} 的标定文件：{calib_path}")

        p2, p3, velo2cam = read_calib(calib_path)
        proj_matrix_2 = p2 @ velo2cam
        proj_matrix_3 = p3 @ velo2cam

        if load_continuous:
            # 连续模式使用左相机的全部图像帧。
            index_pattern = sequence_root / "image_2" / "*.png"
        else:
            # 默认模式以 SSC 体素索引文件确定有效帧，与原 Dataset 保持一致。
            index_pattern = sequence_root / "voxels" / "*.bin"

        frame_paths = sorted(glob.glob(str(index_pattern)))
        for frame_path in tqdm(
            frame_paths,
            desc=f"sequence {sequence}",
            unit="frame",
            dynamic_ncols=True,
        ):
            frame_id = Path(frame_path).stem
            voxel_path = ann_file / sequence / f"{frame_id}_1_1.npy"

            #* 每条记录只保存路径和标定参数，实际数据仍由 __getitem__ 按需读取。
            infos.append(
                {
                    "img_2_path": str(sequence_root / "image_2" / f"{frame_id}.png"),
                    "img_3_path": str(sequence_root / "image_3" / f"{frame_id}.png"),
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "P2": p2,
                    "P3": p3,
                    "T_velo_2_cam": velo2cam,
                    "proj_matrix_2": proj_matrix_2,
                    "proj_matrix_3": proj_matrix_3,
                    # 测试提交集没有公开占用标签，缺失时记录为 None。
                    "voxel_path": str(voxel_path) if voxel_path.is_file() else None,
                }
            )

    return infos


def dump_pickle(infos, output_path, overwrite=False):
    """通过临时文件安全保存 pkl，并默认保护已有结果。"""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} 已存在；如需重新生成，请添加 --overwrite"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(infos, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    ann_file = args.ann_file.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_root / "metadata"
    )

    print(f"数据根目录：{data_root}")
    print(f"标签根目录：{ann_file}")
    print(f"输出目录：{output_dir}")

    #* 只生成三份：本地 test 和 val 都使用 08 序列，应共同读取 val.pkl。
    for split, sequences in SPLITS.items():
        print(f"\n开始生成 {split} metadata")
        infos = create_infos(
            data_root=data_root,
            ann_file=ann_file,
            sequences=sequences,
            load_continuous=args.load_continuous,
        )
        output_path = output_dir / f"semantic_kitti_infos_{split}.pkl"
        dump_pickle(infos, output_path, overwrite=args.overwrite)
        print(f"已保存 {len(infos)} 条样本：{output_path}")


if __name__ == "__main__":
    main()
