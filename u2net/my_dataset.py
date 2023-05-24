import os
os.chdir('/content/finger_u2net')

import cv2
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "train", "train_img")
            self.mask_root = os.path.join(root, "train", "train_label")
        else:
            self.image_root = os.path.join(root, "train", "train_img")
            self.mask_root = os.path.join(root, "train", "train_label")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        #获取所有img以及label的文件路径
        # for p in os.listdir(self.image_root):
        #     if p.endswith(".bmp"):
        #         imfile_path = os.path.join(self.image_root,p)
        #         image_names = [os.path.join(imfile_path,img)for img in os.listdir(imfile_path)  ]
        # for p in os.listdir(self.mask_root):
        #     if p.endswith(".bmp"):
        #         imfile_path = os.path.join(self.image_root, p)
        #         mask_names = [os.path.join(imfile_path, img) for img in os.listdir(imfile_path)]
        f = open(self.image_root + '\\data.txt', 'r')
        image_names = []
        data = f.readlines()
        for line in range(0, len(data)):
            l = data[line].rstrip()
            word = l.split()
            image_names.append(word[0])
        f.close()
        f = open(self.mask_root + '\\data.txt', 'r')
        mask_names = []
        data = f.readlines()
        for line in range(0, len(data)):
            l = data[line].rstrip()
            word = l.split()
            mask_names.append(word[0])
        f.close()
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        # re_mask_names = []
        # for p in image_names:
        #     mask_name = p.replace(".jpg", ".png")
        #     assert mask_name in mask_names, f"{p} has no corresponding mask."
        #     re_mask_names.append(mask_name)
        # mask_names = re_mask_names

        self.images_path = image_names
        self.masks_path = mask_names

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = cv2.imread(image_path, 0)
        assert image is not None, f"failed to read image: {image_path}"
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = Dataset("./", train=True)
    print(len(train_dataset))

    val_dataset = Dataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
