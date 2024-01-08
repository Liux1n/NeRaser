import os
import cv2
import numpy as np
from collections import Counter
from .colmap_read_write_model import read_points3D_binary, read_images_binary

def compute_object_bbox(images_bin_file: str,
                        points3d_bin_file: str,
                        masks_path: str,
                        min_quantile=0.01,
                        max_quantile=0.99,
                        min_num_projections=2,  # require at least 2 feature points to register onto this 3d point
                        ):
    images = read_images_binary(images_bin_file)
    inlier_point3d_ids = Counter()
    for img_id, img in images.items():
        obj_mask = cv2.imread(os.path.join(masks_path, img.name), cv2.IMREAD_GRAYSCALE) != 255
        y, x = obj_mask.shape
        yy, xx = np.meshgrid(range(y), range(x), indexing='ij')
        features_pix_coord = img.xys.astype(np.uint32)
        # since feature coord are rounded, they could go oob; mode="clip" prevents oob access
        features_flattened = np.ravel_multi_index((features_pix_coord[:, 1], features_pix_coord[:, 0]), obj_mask.shape, order='C', mode="clip")
        img_coord = np.array([xx[obj_mask], yy[obj_mask]]).T
        img_coord_falttened = np.ravel_multi_index((img_coord[:, 1], img_coord[:, 0]), obj_mask.shape, order='C') 
        _, inlier_ind, _ = np.intersect1d(features_flattened, img_coord_falttened, return_indices=True)
        inlier_mask = np.zeros(features_pix_coord.shape[0], dtype=bool)
        inlier_mask[inlier_ind] = True
        inlier_point3d_ids.update(img.point3D_ids[inlier_mask])
    inlier_point3d_ids.pop(-1)  # some points are not projected to 3D
    insufficient_projection = [k for k, v in inlier_point3d_ids.items() if v < min_num_projections]
    for k in insufficient_projection:
        del inlier_point3d_ids[k]
    points = read_points3D_binary(points3d_bin_file)
    inlier_point3d = [points[p] for p in inlier_point3d_ids]
    inlier_xyz = np.array([p.xyz for p in inlier_point3d])  # TODO multiply yz by -1 to align with nerfstudio coord sys
    bbox = np.quantile(inlier_xyz, [min_quantile, max_quantile], axis=0)
    return bbox, inlier_point3d, inlier_point3d_ids
