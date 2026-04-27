"""
Occluded-Market-1501 dataset.

Directory layout (root points to the Occluded_Market folder):
  occluded_train/   — occluded training images   (pattern: {pid}_c{cam}s{seq}_{frame}.jpg)
  occluded_query/   — occluded query images       (same pattern)
  bounding_box_test/— Market-1501 gallery images  (pattern: {pid}_c{cam}s{seq}_{frame}_{det}.jpg)

Note: bounding_box_train and query are broken Linux symlinks on Windows; the actual
directories are occluded_train and occluded_query respectively.
"""

import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class OccludedMarket1501(BaseImageDataset):
    """
    Occluded-Market-1501

    Dataset statistics:
      train:   occluded_train  (~9287 images)
      query:   occluded_query  (~2343 images, occluded)
      gallery: bounding_box_test (~15913 images, Market-1501 gallery)
    """

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super().__init__()
        self.dataset_dir = root
        self.train_dir   = osp.join(root, 'occluded_train')
        self.query_dir   = osp.join(root, 'occluded_query')
        self.gallery_dir = osp.join(root, 'bounding_box_test')
        self.pid_begin   = pid_begin

        self._check_before_run()

        train   = self._process_dir(self.train_dir,   relabel=True)
        query   = self._process_dir(self.query_dir,   relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Occluded-Market-1501 loaded")
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

    def _check_before_run(self):
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"Path not found: {d}")

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern   = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid == 0:
                continue  # skip junk (gallery distractors)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid == 0:
                continue
            camid -= 1  # 0-indexed
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset
