from tensorflow.keras.utils import Sequence
import os
import glob
import cv2

from utils import *


class DataLoader(Sequence):
    def __init__(self, batch_size, gamma=2.2, dataset='UCSD', split='train'):
        assert dataset in ['UCSD', 'ICCP']
        assert split in ['train', 'val']
        self.batch_size = batch_size
        self.folders = glob.glob(f'dataset/{dataset}/{split}/*')
        self.gamma = gamma
        self.shuffle()
        self.split = split

    def shuffle(self):
        np.random.shuffle(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index % len(self)]
        exposures = [
            float(line.strip()) for line in
            open(os.path.join(folder, 'input_exp.txt'))
        ]
        hdr_path = os.path.join(folder, 'gt.hdr')
        images = sorted(glob.glob(os.path.join(folder, '*.tif')))

        batch_X_img = []
        batch_Y = []
        batch_hdr = []

        le, me, he = images
        le = cv2.imread(le, cv2.IMREAD_UNCHANGED) / 65535.0
        me = cv2.imread(me, cv2.IMREAD_UNCHANGED) / 65535.0
        he = cv2.imread(he, cv2.IMREAD_UNCHANGED) / 65535.0
        hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)

        le = cv2.cvtColor(le.astype(np.float32), cv2.COLOR_BGR2RGB)
        me = cv2.cvtColor(me.astype(np.float32), cv2.COLOR_BGR2RGB)
        he = cv2.cvtColor(he.astype(np.float32), cv2.COLOR_BGR2RGB)
        hdr = cv2.cvtColor(hdr.astype(np.float32), cv2.COLOR_BGR2RGB)

        le_hdr = ldr_to_hdr(le, exposures[0])
        me_hdr = ldr_to_hdr(me, exposures[1])
        he_hdr = ldr_to_hdr(he, exposures[2])

        patch_size = 128
        for b in range(self.batch_size):
            if self.split == 'train':
                coord_x = np.random.randint(0, hdr.shape[0] - patch_size - 1)
                coord_y = np.random.randint(0, hdr.shape[1] - patch_size - 1)

                le_patch = le[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
                me_patch = me[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
                he_patch = he[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
                le_hdr_patch = le_hdr[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
                me_hdr_patch = me_hdr[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
                he_hdr_patch = he_hdr[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]

                hdr_patch = hdr[coord_x:coord_x + patch_size, coord_y:coord_y + patch_size, :]
            else:
                le_patch, me_patch, he_patch, le_hdr_patch, me_hdr_patch, he_hdr_patch, hdr_patch = \
                    le, me, he, le_hdr, me_hdr, he_hdr, hdr

            x_imgs = [
                np.concatenate([le_patch, le_hdr_patch], axis=-1),
                np.concatenate([me_patch, me_hdr_patch], axis=-1),
                np.concatenate([he_patch, he_hdr_patch], axis=-1),
            ]

            x_imgs = np.stack(x_imgs, axis=0)
            batch_X_img.append(x_imgs)

            y_imgs = hdr_patch
            batch_Y.append(y_imgs)
            batch_hdr.append(hdr_patch)

        ret = [
            np.stack(batch_X_img, axis=0).astype(np.float32),
            np.stack(batch_Y, axis=0).astype(np.float32),
            exposures
        ]
        return ret

    def __len__(self):
        return int(len(self.folders))
