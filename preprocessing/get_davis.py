# run: python get_davis.py <OUT_DIR>
# this file converts the DAVIS dataset into our format.

import os
import shutil
import sys
import subprocess


subprocess.run(['wget', 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'])
subprocess.run(['unzip', 'DAVIS-2017-trainval-480p.zip'])

img_src_root = 'DAVIS/JPEGImages/480p/'
seq_names = os.listdir(img_src_root)
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

for seq_name in seq_names:
    img_src_dir = os.path.join(img_src_root, seq_name)
    img_dst_dir = os.path.join(out_dir, seq_name, 'color')
    shutil.copytree(img_src_dir, img_dst_dir)

    # mask is used only for visualization purposes
    mask_src_root = 'DAVIS/Annotations/480p/'
    mask_src_dir = os.path.join(mask_src_root, seq_name)
    mask_dst_dir = os.path.join(out_dir, seq_name, 'mask')
    shutil.copytree(mask_src_dir, mask_dst_dir)

print('DAVIS data is saved to: {}'.format(os.path.abspath(out_dir)))

