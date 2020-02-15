import numpy
from typing import TypeVar, Callable
narray = TypeVar('numpy.ndarray')

import numpy as np

def fast_euclidean_distance(x:narray, y:narray) -> float:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.linalg.norm(x-y)

def subtractv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"

  return np.subtract(x,y)

def addv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.add(x,y)

def meanv(matrix: narray) -> narray:
  assert isinstance(matrix, numpy.ndarray), f"matrix must be a numpy array but instead is {type(matrix)}"
  assert len(matrix.shape) == 2, f"matrix must be a 2d array but instead is {len(matrix.shape)}d"

  return matrix.mean(axis=0)

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, numpy.ndarray), f"v1 must be a numpy array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, numpy.ndarray), f"v2 must be a numpy array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'

  x = norm(v1)
  if x==0: return 0.0
  y = norm(v2)
  if y==0: return 0.0
  z = x*y
  if z==0: return 0.0  #check for underflow
  return np.dot(v1, v2)/z
