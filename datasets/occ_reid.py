import glob
import os
import os.path as osp

from .bases import BaseImageDataset


class OccludedREID(BaseImageDataset):
    """Occluded-REID dataset.

    200 identities, 5 occluded query images + 5 whole-body gallery images each.
    Evaluation only — no training split.

    Directory structure:
        <root>/
            occluded_body_images/   <- query  (cam_id = 0)
                001/  *.tif
                002/  *.tif
                ...
            whole_body_images/      <- gallery (cam_id = 1)
                001/  *.tif
                ...
    """

    def __init__(self, root='', verbose=True, **kwargs):
        super().__init__()
        self.query_dir   = osp.join(root, 'occluded_body_images')
        self.gallery_dir = osp.join(root, 'whole_body_images')

        if not osp.exists(self.query_dir):
            raise RuntimeError(f"Query dir not found: {self.query_dir}")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"Gallery dir not found: {self.gallery_dir}")

        self.train = []
        query   = self._process_dir(self.query_dir,   cam_id=0)
        gallery = self._process_dir(self.gallery_dir, cam_id=1)

        if verbose:
            print("=> Occluded-REID loaded")
            self.print_dataset_statistics(self.train, query, gallery)

        self.query   = query
        self.gallery = gallery

        self.num_train_pids  = 0
        self.num_train_imgs  = 0
        self.num_train_cams  = 0
        self.num_train_vids  = 0
        self.num_query_pids,   self.num_query_imgs,   self.num_query_cams,   _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, cam_id):
        subdirs = sorted([d for d in os.listdir(dir_path)
                          if osp.isdir(osp.join(dir_path, d))])
        pid_map = {name: idx for idx, name in enumerate(subdirs)}

        dataset = []
        for subdir in subdirs:
            pid = pid_map[subdir]
            for img_path in sorted(glob.glob(osp.join(dir_path, subdir, '*'))):
                if img_path.lower().endswith(('.tif', '.jpg', '.png', '.jpeg')):
                    dataset.append((img_path, pid, cam_id, 1))
        return dataset
