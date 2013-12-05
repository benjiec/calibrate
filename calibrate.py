from scipy import stats


class LinearCurve(object):
  """
  Constructs a linear curve using x and y values from standards, and backout
  the x value for an unknown based on its y signal. Assumes x and y values are
  in the linear region of a calibration curve.
  """

  def __init__(self, x, y):
    self.__x = x
    self.__y = y
    self.__m, self.__b, self.__r_value, self.__p_value, self.__stderr = stats.linregress(x, y)

  @property
  def slope(self):
    """
    Returns slope of the curve.

    >>> LinearCurve([1,2,3],[3,6,9]).slope
    3.0
    >>> LinearCurve([1,2,3],[9,6,3]).slope
    -3.0
    """
    return self.__m

  @property
  def y_intercept(self):
    """
    Returns y-intercept of the curve.

    >>> LinearCurve([1,2,3],[5,8,11]).y_intercept
    2.0
    >>> LinearCurve([1,2,3],[1,4,7]).y_intercept
    -2.0
    """
    return self.__b

  @property
  def r_squared(self):
    """
    Returns the R^2 value of the curve. R is the correlation between the x and
    y values.

    >>> LinearCurve([1,2,3],[3,6,9]).r_squared
    1.0
    >>> LinearCurve([1,2,3],[2,6,7]).r_squared
    0.89285714285714279
    >>> LinearCurve([1,2,3],[5,5,5]).r_squared
    0.0
    >>> LinearCurve([1,2,2,3],[3,6,5,9]).r_squared
    0.96000000000000019
    """
    return self.__r_value**2

  def interpolate(self, y):
    """
    Find x value corresponding to the y value using the curve. Returns a bound
    of [x_lo, x_hi], where x_hi-x_lo is the error of interpolation. If y lies
    below or above the limits of quantification or linearity, x_lo or x_hi may
    be None.
  
    >>> c = LinearCurve([1,2,3],[3,6,9])
    >>> c.interpolate(7.5)
    2.5
    """

    # XXX compute error
    # See http://www.chem.utoronto.ca/coursenotes/analsci/StatsTutorial/ConcCalib.html
    # 
    # XXX find limits of quantification and linearity and adjust results based
    # on that.

    x = (y-self.y_intercept)*1.0/self.slope
    return x
    

class CalibrationCurve(object):
  """
  Constructs a calibration curve using x and y values from standards. Finds the
  linear region of the calibration curve and limits of linearity.
  """
  
  def __init__(self, x, y):
    self.__x = x
    self.__y = y

  def linear_region(self):
    """
    Returns the linear region of the calibration curve.

    >>> CalibrationCurve([1,2,3],[2,4,6]).linear_region().slope
    2.0
    """

    # XXX computes limits of linearity
    return LinearCurve(self.__x, self.__y)



if __name__ == '__main__':
  import doctest
  doctest.testmod()

