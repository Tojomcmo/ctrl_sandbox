from jax.numpy import ndarray
from numpy import float64, complex128, complex64
import numpy.typing as npt
from typing import Callable, Tuple, Union

npArr64 = npt.NDArray[float64]
npCpx128 = npt.NDArray[complex128]
npCpx64 = npt.NDArray[complex64]

JaxArrayTuple2 = Tuple[ndarray, ndarray]
JaxArrayTuple3 = Tuple[ndarray, ndarray, ndarray]
JaxArrayTuple4 = Tuple[ndarray, ndarray, ndarray, ndarray]
JaxArrayTuple5 = Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]

npORjnpArr = Union[ndarray, npt.NDArray]

DynFuncType = Callable[[ndarray, ndarray], ndarray]

CostFuncType = Callable[[ndarray, ndarray, int], JaxArrayTuple3]
CostFuncDiffType = Callable[[ndarray, ndarray, int], ndarray]
CostFuncFloatType = Callable[[ndarray, ndarray, int], Tuple[float, float, float]]

IndexJax2JaxFuncType = Callable[[int, ndarray], ndarray]
