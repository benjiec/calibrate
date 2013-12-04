from scipy import stats

class CalibrationCurve(object):
  """
  Constructs a calibration curve using concentrations or whatever (x-values)
  and signals (y-values) from standards, and backout the concentration of an
  unknown based on its signal. Computes quality of calibration curve and
  estimation error.

  See http://www.chem.utoronto.ca/coursenotes/analsci/StatsTutorial/ConcCalib.html
  """

  def __init__(self, x, y):
    """
    Create a calibration curve object with just x and y values. x and y should
    be array of the same length.
    """

    self.__x = x
    self.__y = y
    self.__m, self.__b, self.__r_value, self.__p_value, self.__stderr = stats.linregress(x, y)

  @property
  def slope(self):
    """
    >>> CalibrationCurve([1,2,3],[3,6,9]).slope
    3.0
    >>> CalibrationCurve([1,2,3],[9,6,3]).slope
    -3.0
    """
    return self.__m

  @property
  def y_intercept(self):
    """
    >>> CalibrationCurve([1,2,3],[5,8,11]).y_intercept
    2.0
    >>> CalibrationCurve([1,2,3],[1,4,7]).y_intercept
    -2.0
    """
    return self.__b

  @property
  def r_squared(self):
    """
    >>> CalibrationCurve([1,2,3],[3,6,9]).r_squared
    1.0
    >>> CalibrationCurve([1,2,3],[2,6,7]).r_squared
    0.89285714285714279
    >>> CalibrationCurve([1,2,3],[5,5,5]).r_squared
    0.0
    """
    return self.__r_value**2


#import numpy as np
#c = CalibrationCurve(np.random.random(10), np.random.random(10))
#print c.r_squared

if __name__ == '__main__':
  import doctest
  doctest.testmod()

