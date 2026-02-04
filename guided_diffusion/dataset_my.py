import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from skimage.feature import canny
from skimage.color import rgb2gray
import os
import linecache
def get_caption_from_path(path: str,
                          file_a='/mnt/data/mzy/25sem_vis_gaijin2_dh/train_mask_caption_zong.txt'):
    """
    根据图片路径与两份 caption 文件获取 caption 文本。
    path: 图片完整路径，如 /mnt/data/mzy/guidediff/data2/train_28000/08817.jpg
    file_a: 文本文件 a 的路径
    file_b: 文本文件 b 的路径
    返回: (caption_str, idx, which_file)
        caption_str: 取到的 caption（去掉结尾换行）
        idx: 实际读取的行号（从 1 开始）
        which_file: 'a' 或 'b' 或 None
    """
    fname = os.path.splitext(os.path.basename(path))[0]
    try:
        num = int(fname)
    except ValueError:
        raise ValueError(f"无法从文件名 '{fname}' 提取整数 ID")

    line_idx = num + 1
    which = 'a'
    caption = linecache.getline(file_a, line_idx)
    caption = caption.rstrip('\n\r')
    if caption == "":
        raise IndexError(f"在文件 {which} 的第 {line_idx} 行没有读到内容，请检查文件长度。")

    return caption, line_idx, which
def get_caption_from_path1(path: str,
                          file_a='/mnt/data/mzy/25sem_vis_gaijin2_dh/gaijin/test_captionxr1_23.txt'):
    """
    根据图片路径与两份 caption 文件获取 caption 文本。
    path: 图片完整路径，如 /mnt/data/wxX/guidediff/data2/train_28000/08817.jpg
    file_a: 文本文件 a 的路径
    file_b: 文本文件 b 的路径
    返回: (caption_str, idx, which_file)
        caption_str: 取到的 caption（去掉结尾换行）
        idx: 实际读取的行号（从 1 开始）
        which_file: 'a' 或 'b' 或 None
    """
    fname = os.path.splitext(os.path.basename(path))[0]
    try:
        num = int(fname)
    except ValueError:
        raise ValueError(f"无法从文件名 '{fname}' 提取整数 ID")
    line_idx = num + 1
    which = 'a'
    caption = linecache.getline(file_a, line_idx)
    caption = caption.rstrip('\n\r')
    if caption == "":
        raise IndexError(f"在文件 {which} 的第 {line_idx} 行没有读到内容，请检查文件长度。")

    return caption, line_idx, which


def load_data(
    *,
    data_dir,
    mask_dir,
    tokenizer,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    mask_train=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    mask_files = _list_image_files_recursively(mask_dir)
    classes = None
    if class_cond:


        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        mask_files,
        tokenizer,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        mask_train=mask_train
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        mask_paths,
        tokenizer,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        mask_train=True,

    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        print("len(images):",len(image_paths))
        self.mask_paths=mask_paths[shard:][::num_shards]
        print("len(masks):",len(mask_paths))
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.mask_train=mask_train
        self.tokenizer = tokenizer

        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])


        print("初始化成功！")

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):

        print("原图idx：", idx)

        path = self.local_images[idx]
        print("path:",path)


        caption, idx, which = get_caption_from_path1(path)
        print("caption:",caption)

        if self.tokenizer is None:
            raise RuntimeError("ImageDataset 需要传入 tokenizer 才能在 dataset 内分词。")

        tx = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        caption_input_ids = tx["input_ids"].squeeze(0)
        caption_attention_mask = tx["attention_mask"].squeeze(0)



        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image1 = pil_image
        pil_image = pil_image.convert("RGB")

        image_gray = pil_image1.convert("L")
        image_gray1 = image_gray

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
            gray_arr = random_crop_arr(image_gray, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
            gray_arr = center_crop_arr(image_gray, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            gray_arr = gray_arr[:, ::-1]

        arr1 = arr
        gray_arr1 = gray_arr

        arr = arr.astype(np.float32) / 127.5 - 1


        arr_sam = arr1.astype(np.float32) / 255.
        arr_sam = arr_sam - self.imagenet_mean
        arr_sam = arr_sam / self.imagenet_std
        gray_arr = gray_arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        if self.mask_train == True:
            mask_idx = random.randint(0, len(self.mask_paths) - 1)
        else:
            mask_idx = idx-1
        print("mask的idx：", mask_idx)
        maskpath = self.mask_paths[mask_idx]
        print("maskpath:",maskpath)
        with bf.BlobFile(maskpath, "rb") as f:
            pil_maskimage = Image.open(f)
            pil_maskimage.load()
        pil_maskimage = pil_maskimage.convert("L")

        mask_arr = center_crop_arr(pil_maskimage, 256)
        mask_arr = (mask_arr > 0).astype(np.uint8) * 255

        mask_arr1 = mask_arr
        mask_arr77 = mask_arr


        mask_arr_sam=mask_arr77.astype(np.float32)/255
        mask_arr_sam= np.expand_dims(mask_arr_sam, axis=2)
        mask_arr = mask_arr.astype(np.float32) / 127.5 - 1
        mask_arr3 = mask_arr
        mask_arr = np.expand_dims(mask_arr, axis=2)


        defect_arr = (mask_arr == -1).astype(int) * arr



        image_edge = canny(gray_arr1, sigma=1.9).astype(np.float_)

        image_edge = image_edge * 2 - 1
        image_edge = np.expand_dims(image_edge, axis=2)




        mask_arr2 = (1 - mask_arr1 / 255).astype(np.bool_)
        quesun_edge = canny(gray_arr1, sigma=1.9, mask=mask_arr2).astype(np.float_)
        quesun_edge = quesun_edge * 2 - 1
        quesun_edge1 = np.expand_dims(quesun_edge, axis=2)



        '''
        gray_mask_image = rgb2gray(defect_arr)

        gray_mask_image = np.expand_dims(gray_mask_image, axis=2)
        '''

        mask_gray_image = (mask_arr3 == -1).astype(int) * gray_arr
        mask_gray_image = np.expand_dims(mask_gray_image, axis=2)


        gray_image_arr = np.expand_dims(gray_arr, axis=2)




        print("下面五个shape分别是原图、mask、缺损图、原图边缘图、缺损边缘图、缺损灰度图、完整灰度图的shape：")
        print(arr.shape)
        print(mask_arr.shape)
        print(defect_arr.shape)
        print(image_edge.shape)
        print(quesun_edge1.shape)
        print(mask_gray_image.shape)
        print(gray_image_arr.shape)














        return np.transpose(arr, [2, 0, 1]), np.transpose(mask_arr, [2, 0, 1]),\
               np.transpose(defect_arr,[2, 0, 1]), np.transpose(image_edge, [2, 0, 1]),\
               np.transpose(quesun_edge1, [2, 0, 1]), np.transpose(mask_gray_image,[2, 0, 1]),\
               np.transpose(gray_image_arr, [2, 0, 1]), \
               np.transpose(arr_sam, [2, 0, 1]),np.transpose(mask_arr_sam, [2, 0, 1]),\
               caption,\
               caption_input_ids,\
               caption_attention_mask,\
               out_dict


def center_crop_arr(pil_image, image_size):



    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)


    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
