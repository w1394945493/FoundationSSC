import os
import glob
import numpy as np
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class KITTI360Dataset(Dataset):
    def __init__(
        self,
        data_root,
        ann_file,
        pipeline,
        split,
        camera_used,
        occ_size,
        pc_range,
        test_mode=False,
        load_continuous=False,
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.splits = {
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

        self.sequences = self.splits[split]

        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(self.ann_file)

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {"left": "2", "right": "3"}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print("found None in training data")
            return None

        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print("found None in training data")
            return None

        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_data_info(self, index):
        info = self.data_infos[index]
        """
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        """
        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info["sequence"],
            frame_id=info["frame_id"],
        )

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info["img_{}_path".format(int(cam_type))])
            lidar2img_rts.append(info["proj_matrix_{}".format(int(cam_type))])
            cam_intrinsics.append(info["P{}".format(int(cam_type))])
            lidar2cam_rts.append(info["T_velo_2_cam"])

        focal_length = info["P2"][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline,
            )
        )
        # gt_occ is None for test-set
        input_dict["gt_occ"] = self.get_ann_info(index, key="voxel_path")

        return input_dict

    def load_annotations(self, ann_file=None):
        """扫描当前数据划分中的文件，生成每一帧样本的元信息列表。

        注意：参数 ``ann_file`` 当前没有在函数内部使用，实际标签根目录取自
        ``self.ann_file``。本函数只收集路径和标定信息；除检查占用标签是否存在
        外，不会在这里真正读取图像、点云或占用标签内容。
        """
        # 保存全部样本的元信息；列表中的每个字典对应一个帧。
        scans = []

        # self.sequences 由 train/val/test 划分决定，逐个扫描当前划分包含的序列。
        for sequence in self.sequences:
            # KITTI-360 在这里使用固定标定参数：P2/P3 是左右相机投影矩阵，
            # Tr 将激光雷达坐标转换到相机坐标。
            calib = self.read_calib()
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]

            # 组合得到从激光雷达坐标直接投影到左右图像平面的矩阵。
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            # 占用标签目录和当前序列的原始图像目录。
            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "data_2d_raw", sequence)

            if self.load_continuous:
                # 连续帧模式：直接以左相机目录下的所有 PNG 图像作为样本索引。
                id_base_path = os.path.join(
                    self.data_root,
                    "data_2d_raw",
                    sequence,
                    "image_00",
                    "data_rect",
                    "*.png",
                )
            else:
                # 默认模式：以 voxels 目录下已有的 .bin 文件确定需要评估的帧。
                id_base_path = os.path.join(
                    self.data_root, "data_2d_raw", sequence, "voxels", "*.bin"
                )

            # 展开通配符并排序，保证每次构建数据集时样本顺序稳定。
            all_id_base_path = sorted(glob.glob(id_base_path))
            for id_path in all_id_base_path:
                # 从文件名中取得不带扩展名的帧编号，例如 0000001234。
                img_id = id_path.split("/")[-1].split(".")[0]

                # 根据帧编号拼出左右相机图像和占用标签的完整路径。
                img_2_path = os.path.join(
                    img_base_path, "image_00", "data_rect", img_id + ".png"
                )
                img_3_path = os.path.join(
                    img_base_path, "image_01", "data_rect", img_id + ".png"
                )
                voxel_path = os.path.join(voxel_base_path, img_id + "_1_1.npy")

                # 测试集可能没有真实占用标签。用 None 标记后，后续 pipeline
                # 会在测试阶段生成占位标签，而不是在这里读取不存在的文件。
                if not os.path.exists(voxel_path):
                    voxel_path = None

                # 汇总该帧后续数据 pipeline 所需的路径、序列信息和标定矩阵。
                # 图像及标签内容会在 __getitem__ 被调用时才真正加载。
                scans.append(
                    {
                        "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                    }
                )

        # 返回当前数据划分的完整样本索引，赋给 self.data_infos。
        return scans

    def get_ann_info(self, index, key="voxel_path"):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)

    @staticmethod
    def read_calib(calib_path=None):
        """
        Tr transforms a point from velodyne coordinates into the
        left rectified camera coordinate system.
        In order to map a point X from the velodyne scanner to a
        point x in the i'th image plane, you thus have to transform it like:
        x = Pi * Tr * X
        """
        P2 = np.array(
            [
                [552.554261, 0.000000, 682.049453, 0.000000],
                [0.000000, 552.554261, 238.769549, 0.000000],
                [0.000000, 0.000000, 1.000000, 0.000000],
            ]
        ).reshape(3, 4)

        P3 = np.array(
            [
                [552.554261, 0.000000, 682.049453, -328.318735],
                [0.000000, 552.554261, 238.769549, 0.000000],
                [0.000000, 0.000000, 1.000000, 0.000000],
            ]
        ).reshape(3, 4)

        cam2velo = np.array(
            [
                [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                [0, 0, 0, 1],
            ]
        ).reshape(4, 4)

        velo2cam = np.linalg.inv(cam2velo)
        calib_out = {}
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)
        calib_out["P2"][:3, :4] = P2.reshape(3, 4)
        calib_out["P3"][:3, :4] = P3.reshape(3, 4)
        calib_out["Tr"] = np.identity(4)
        calib_out["Tr"][:3, :4] = velo2cam[:3, :4]
        return calib_out

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def decompose_projection_matrix(self, p):
        """
        Shortcut to use cv2.decomposeProjectionMatrix(), which only returns k, r, t, and divides
        t by the scale, then returns it as a vector with shape (3,) (non-homogeneous)

        Arguments:
        p -- projection matrix to be decomposed

        Returns:
        k, r, t -- intrinsic matrix, rotation matrix, and 3D translation vector

        """
        import cv2

        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
        t = (t / t[3])[:3]

        return k, r, t

    def dynamic_baseline(self, infos):
        P3 = infos["P3"]
        P2 = infos["P2"]
        # baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        k_left, r_left, t_left = self.decompose_projection_matrix(P2[:3, ...])
        k_right, r_right, t_right = self.decompose_projection_matrix(P3[:3, ...])
        baseline = t_right[0] - t_left[0]
        return baseline
