import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Comp5421Segmentation(Dataset):
    NUM_CLASSES = 7
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('comp5421'),
                 split='train'):
        super().__init__()
        ids_file = os.path.join(base_dir, '{}/{}.txt'.format(split, split))
        self.ids = []
        with open(ids_file, 'r') as f:
            for i, l in enumerate(f):
                self.ids.append(l.rsplit()[0])

        self.img_dir = os.path.join(base_dir, '{}/images'.format(split))
        self.label_dir = os.path.join(base_dir, '{}/labels'.format(split))
        self.split = split
        self.args = args

    def __getitem__(self, index):
        _img, _target, _img_id = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'imgId': _img_id}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        img_id = self.ids[index]
        
        _img = Image.open(os.path.join(self.img_dir, "{}.png".format(img_id))).convert('RGB')
        _target = Image.open(os.path.join(self.label_dir, "{}.png".format(img_id))).convert('L')

        return _img, _target, img_id

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed = composed_transforms(sample)
        transformed['imgId'] = sample['imgId']
        transformed['resolution'] = sample['image'].size

        return transformed


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    comp5421_val = Comp5421Segmentation(args, split='val')

    dataloader = DataLoader(comp5421_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='comp5421')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)