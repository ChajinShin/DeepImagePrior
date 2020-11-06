import numpy as np
import skimage.transform as transform
import torch
from .common_types import NumpyArray, TensorType, _ndarray_tensor, _size_2_t, _value_2_or_3_t


def _resize(img: NumpyArray, size: _size_2_t) -> NumpyArray:
    # img shape 확인하여 이미지인지 아닌지 판단
    if (len(img.shape) != 2) and (len(img.shape) != 3):
        raise ValueError("Input '{}' is not a image shape. H, W or H, W, C shape is available.")

    # 만약 img_size가 tuple이 아닌 상수라면 정사각형으로 취급한다.
    if isinstance(size, int):
        resize_size = [size, size]
    else:
        resize_size = size

    resized_img = transform.resize(img, resize_size, anti_aliasing=True)
    return resized_img


def _normalize(img: NumpyArray, mean: _value_2_or_3_t, std: _value_2_or_3_t, eps=1e-8) -> NumpyArray:
    # 2차원 이미지 처리
    if len(img.shape) == 2:
        if isinstance(mean, float):
            mean = [mean]
        if isinstance(std, float):
            std = [std]
        mean = np.array(mean)
        std = np.array(std)

        # 2차원 이미지는 mean, std 의 차원은 1차원이어야 한다.
        if (mean.shape[-1] != 1) or (std.shape[-1] != 1):
            raise ValueError("'mean' or 'std' dimension have to be one with H, W type image dimension")

    # ----------------------------------------
    # 3차원 이미지 처리
    elif len(img.shape) == 3:
        # mean, value 처리
        if isinstance(mean, float):
            mean = [mean] * 3
        if isinstance(std, float):
            std = [std] * 3

        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        if (mean.shape[-1] != 3) or (std.shape[-1] != 3):
            raise ValueError("'mean' or 'std' dimension is not matched with H, W, C type image dimension")

    # -------------------------------------
    else:
        raise ValueError("'img' dimension is {}. Only 2-dimensional or 3-dimensional image is supported".format(len(img.shape)))

    # normalization
    img = (img - mean) / (std + eps)
    img = np.ascontiguousarray(img)
    return img


def _to_tensor(img: NumpyArray) -> TensorType:
    # 2차원 이미지라면 3차원의 1차원 채널을 가지도록 바꾼다.
    if len(img.shape) == 2:
        img = img[np.newaxis, ...]
    elif len(img.shape) == 3:
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
    else:
        raise ValueError("'img' dimension is {}. ONly 2-dimensional or 3-dimensional image is supported.".format(len(img.shape)))
    return torch.from_numpy(img)


# --------------------------------------------------------------
def simple_denorm(img: _ndarray_tensor, shift: float = 1, scale: float = 2, min: float = 0, max: float = 1):
    out = (img + shift) / scale
    if type(img) == NumpyArray:
        out = np.clip(out, min, max)
    elif type(img) == TensorType:
        out = out.clamp(min, max)
    else:
        raise ValueError("input img type is not a ndarray or Tensor.")
    return out


# --------------------------------------------------------------
class ProcessingUnit(object):
    def __init__(self, *process_list):
        self.process_list = process_list

    def __call__(self, img: NumpyArray):
        for process in self.process_list:
            img = process(img)
        return img


class Resize(object):
    def __init__(self, size: _size_2_t):
        self.size = size

    def __call__(self, img: NumpyArray) -> NumpyArray:
        return _resize(img, self.size)


class Normalize(object):
    def __init__(self, mean: _value_2_or_3_t, std: _value_2_or_3_t, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, img: NumpyArray) -> NumpyArray:
        return _normalize(img, self.mean, self.std, self.eps)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img: NumpyArray) -> TensorType:
        return _to_tensor(img)

