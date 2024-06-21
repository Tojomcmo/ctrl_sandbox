from jax.numpy import ndarray
import numpy.typing as npt
from typing import Callable, Tuple, Union

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
