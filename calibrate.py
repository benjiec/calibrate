from scipy import stats


class BaseCurve(object):

  def __init__(self, x, y):
    points = sorted(zip(x, y), key=lambda p: p[0])
    self.x = tuple([p[0] for p in points])
    self.y = tuple([p[1] for p in points])


class LinearCurve(BaseCurve):
  """
  Constructs a linear curve with known x and y values, and interpolates the x
  value for an unknown based on the unknown's y signal. Initial x and y values
  must have a linear relationship.
  """

  def __init__(self, x, y):
    super(LinearCurve, self).__init__(x, y)
    self.__m, self.__b, self.__r_value, self.__p_value, self.__stderr = stats.linregress(self.x, self.y)

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


class CalibrationCurve(BaseCurve):
  """
  Constructs a calibration curve using x and y values from standards. Finds the
  linear region of the calibration curve and limits of linearity.

  Assumes the input x and y values looks like a step function: initial flat
  line at the start of the curve representing areas near limits of detection,
  then a linear curve representing the dynamic range, and then another flat
  line representing limits of linearity and signal saturation.
  """

  def __init__(self, x, y):
    super(CalibrationCurve, self).__init__(x, y)
    self.__linear_region = None
    self.__lol = None

  def linear_region(self):
    """
    Returns the linear region of the calibration curve.

    >>> l = CalibrationCurve([1,2,3],[2,4,6]).linear_region()
    >>> zip(l.x, l.y)
    [(1, 2), (2, 4), (3, 6)]

    >>> l = CalibrationCurve([1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]).linear_region()
    >>> zip(l.x, l.y)
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]

    >>> l = CalibrationCurve([1,1,3,3,4,5,6,7,8],[2,2,2,2,4,6,10,8,8]).linear_region()
    >>> zip(l.x, l.y)
    [(3, 2), (4, 4), (5, 6)]

    >>> l = CalibrationCurve([1,2,3,4],[2,2,2,2]).linear_region()
    >>> l is None
    True
    """

    if self.__linear_region is not None:
      return self.__linear_region

    ELBOW_R = 0.6
    linear_pos = []

    # find first elbow
    for i in range(2, len(self.x)):
      x = self.x[0:i]
      y = self.y[0:i]
      r = stats.pearsonr(x,y)[0]
      if r is not None and r >= ELBOW_R:
        linear_pos.append(i-2)
        break

    if len(linear_pos) == 0: # cannot find start of linear region, or just bad data!
      return None

    # find second elbow
    for i in range(2, len(self.x)-linear_pos[0]):
      x = self.x[len(self.x)-i:len(self.x)]
      y = self.y[len(self.x)-i:len(self.x)]
      r = stats.pearsonr(x,y)[0]
      if r is not None and r >= ELBOW_R:
        linear_pos.append(len(self.x)-i+1)
        break

    if len(linear_pos) == 1: # all linear
      linear_pos.append(len(self.x)-1)

    self.__lol = tuple(linear_pos)
    self.__linear_region = LinearCurve(self.x[linear_pos[0]:linear_pos[1]+1],
                                       self.y[linear_pos[0]:linear_pos[1]+1])
    return self.__linear_region

  def lol(self):
    """
    Returns limits of linearity.

    >>> CalibrationCurve([1,2,3],[2,4,6]).lol()
    (0, 2)

    >>> CalibrationCurve([1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]).lol()
    (0, 7)

    >>> CalibrationCurve([1,1,3,3,4,5,6,7,8],[2,2,2,2,4,6,10,8,8]).lol()
    (3, 5)

    >>> CalibrationCurve([1,2,3,4],[2,2,2,2]).lol() is None
    True
    """

    if self.__lol is None:
      self.linear_region()
    return self.__lol


if __name__ == '__main__':
  import doctest
  doctest.testmod()

