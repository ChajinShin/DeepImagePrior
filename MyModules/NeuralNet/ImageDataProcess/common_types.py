from typing import Union, Tuple
from numpy import ndarray
from torch import Tensor

# 유용한 type 형식 지정
NumpyArray = ndarray
TensorType = Tensor

# int 사용한 특정 size 표현
_size_any_t = Union[int, Tuple[int, ...]]
_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]

# float 사용한 특정 값 표현
_value_any_t = Union[float, Tuple[float, ...]]
_value_1_t = Union[float, Tuple[float]]
_value_2_t = Union[float, Tuple[float, float]]
_value_3_t = Union[float, Tuple[float, float, float]]

_value_2_or_3_t = Union[float, Tuple[float, float], Tuple[float, float, float]]

# ndarray and tensor mix
_ndarray_tensor = Union[TensorType, NumpyArray]

