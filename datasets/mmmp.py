import glob
import re
import os
import os.path as osp
import random
from collections import defaultdict

from .bases import BaseImageDataset


class MMMP(BaseImageDataset):
    dataset_dir = '/data/mmmp1_10'

    def __init__(self, root='', verbose=True, pid_begin=0, exp_setting=None, split_root=None, **kwargs):
        super(MMMP, self).__init__()
        if root:
            self.dataset_dir = root
        self.exp_setting       = exp_setting
        self.setting_name_split = exp_setting.split("_")
        split_dir              = split_root if split_root else self.dataset_dir
        self.file_path_train   = osp.join(split_dir, self.exp_setting, 'train_id.txt')
        self.file_path_val     = osp.join(split_dir, self.exp_setting, 'val_id.txt')
        self.file_path         = osp.join(split_dir, self.exp_setting, 'test_id.txt')
        self.pid_begin         = pid_begin
        self.split_ratio       = 0.5

        if len(self.setting_name_split) == 2:   # e.g. exp_rgb
            train          = self._process_train(self.dataset_dir, self.file_path_train,
                                                 self.file_path_val, self.exp_setting, relabel=True)
            query, gallery = self._process_same(self.dataset_dir, self.file_path,
                                                self.exp_setting, relabel=False,
                                                split_ratio=self.split_ratio)
        elif len(self.setting_name_split) == 5:  # e.g. exp_cctv_ir_cctv_rgb
            train   = self._process_train(self.dataset_dir, self.file_path_train,
                                          self.file_path_val, self.exp_setting, relabel=True)
            query   = self._process_query(self.dataset_dir, self.file_path,
                                          self.exp_setting, relabel=False)
            gallery = self._process_gallery(self.dataset_dir, self.file_path,
                                            self.exp_setting, relabel=False)

        if verbose:
            print("=> MMMP loaded")
            self.print_dataset_statistics(train, query, gallery)
        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = \
            self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cameras_for_setting(setting_name_split, slot_start):
        """Return camera list for the modality/platform at index slot_start."""
        platform = setting_name_split[slot_start]
        modality = setting_name_split[slot_start + 1]
        if platform == 'cctv' and modality == 'ir':
            return ['07', '08', '09', '10', '11', '12']
        if platform == 'cctv' and modality == 'rgb':
            return ['01', '02', '03', '04', '05', '06']
        if platform == 'uav' and modality == 'ir':
            return ['14']
        if platform == 'uav' and modality == 'rgb':
            return ['13']
        raise ValueError(f"Unknown platform/modality: {platform}/{modality}")

    @staticmethod
    def _cameras_for_single(setting_name_split):
        """Return camera list for a single-modality exp setting (len==2)."""
        modality = setting_name_split[1]
        if modality == 'cctv':
            return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        if modality == 'uav':
            return ['13', '14']
        if modality == 'ir':
            return ['07', '08', '09', '10', '11', '12', '14']
        if modality == 'rgb':
            return ['01', '02', '03', '04', '05', '06', '13']
        raise ValueError(f"Unknown modality: {modality}")

    @staticmethod
    def _list_jpgs(img_dir):
        return sorted([img_dir + '/' + f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    @staticmethod
    def _parse_path(img_path):
        """Extract (camid, pid) from a path of the form .../CC/PPPP/XXXXXXXX.jpg"""
        camid = int(img_path[-15])
        pid   = int(img_path[-13:-9])
        return camid, pid

    def _build_dataset(self, files, relabel, pid2label=None):
        if pid2label is None:
            pids      = {self._parse_path(p)[1] for p in files}
            pid2label = {pid: label for label, pid in enumerate(sorted(pids))}
        dataset = []
        for img_path in files:
            camid, pid = self._parse_path(img_path)
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset

    def _collect_files(self, dir_path, ids, cameras):
        files = []
        for pid in sorted(ids):
            for cam in cameras:
                img_dir = osp.join(dir_path, cam, pid)
                if osp.isdir(img_dir):
                    files.extend(self._list_jpgs(img_dir))
        return files

    # ------------------------------------------------------------------
    # Public processing methods
    # ------------------------------------------------------------------

    def _process_train(self, dir_path, file_path_train, file_path_val,
                       exp_setting=None, relabel=True):
        with open(file_path_train, 'r') as f:
            id_train = ["%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        with open(file_path_val, 'r') as f:
            id_val   = ["%04d" % int(x) for x in f.read().splitlines()[0].split(',')]

        s = exp_setting.split("_")
        if len(s) == 5:
            cameras = self._cameras_for_setting(s, 1) + self._cameras_for_setting(s, 3)
        else:
            cameras = self._cameras_for_single(s)

        files = self._collect_files(dir_path, id_train + id_val, cameras)
        return self._build_dataset(files, relabel)

    def _process_query(self, dir_path, file_path, exp_setting=None, relabel=False):
        with open(file_path, 'r') as f:
            ids = ["%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        s = exp_setting.split("_")
        cameras = self._cameras_for_setting(s, 1)
        files   = self._collect_files(dir_path, ids, cameras)
        return self._build_dataset(files, relabel)

    def _process_gallery(self, dir_path, file_path, exp_setting=None, relabel=False):
        with open(file_path, 'r') as f:
            ids = ["%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        s = exp_setting.split("_")
        cameras = self._cameras_for_setting(s, 3)
        files   = self._collect_files(dir_path, ids, cameras)
        return self._build_dataset(files, relabel)

    def _process_same(self, dir_path, file_path, exp_setting=None,
                      relabel=False, split_ratio=0.5):
        with open(file_path, 'r') as f:
            ids = ["%04d" % int(x) for x in f.read().splitlines()[0].split(',')]
        s       = exp_setting.split("_")
        cameras = self._cameras_for_single(s)
        all_files = self._collect_files(dir_path, ids, cameras)

        id_cam_files = defaultdict(list)
        for img_path in all_files:
            camid, pid = self._parse_path(img_path)
            id_cam_files[(pid, camid)].append(img_path)

        pids      = {k[0] for k in id_cam_files}
        pid2label = {pid: label for label, pid in enumerate(sorted(pids))}

        query_files, gallery_files = [], []
        for (pid, camid), files in id_cam_files.items():
            if len(files) == 1:
                continue
            random_files  = files.copy()
            random.shuffle(random_files)
            split_point   = max(1, int(len(files) * split_ratio))
            labeled_pid   = pid2label[pid] if relabel else pid
            for img_path in random_files[:split_point]:
                query_files.append((img_path, self.pid_begin + labeled_pid, camid, 0))
            for img_path in random_files[split_point:]:
                gallery_files.append((img_path, self.pid_begin + labeled_pid, camid, 0))

        return query_files, gallery_files
