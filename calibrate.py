from scipy import stats
import math
import numpy as np


class BaseCurve(object):

  def __init__(self, x, y):
    """
    Constructor for base class for curve classes. Sorts input data by x value.

    >>> c = BaseCurve([2,3,1,4],[1,2,3,4])
    >>> zip(c.x, c.y)
    [(1, 3), (2, 1), (3, 2), (4, 4)]
    """

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
    """
    Constructs a linear curve.

    >>> LinearCurve([1,2,3],[3,6,9]).slope
    3.0
    >>> LinearCurve([1,2,3],[5,8,11]).y_intercept
    2.0
    >>> LinearCurve([1,2,3],[1,4,7]).y_intercept
    -2.0
    >>> LinearCurve([1,2,3],[3,6,9]).s_y
    0.0

    >>> x = np.random.random(10)
    >>> y = np.random.random(10)
    >>> round(LinearCurve(x, y).r_value, 6) == round(stats.pearsonr(x, y)[0], 6)
    True

    >>> x = np.random.random(10)
    >>> y = np.random.random(10)
    >>> l = LinearCurve(x, y)
    >>> s_y = math.sqrt(sum([(yi-l.slope*xi-l.y_intercept)**2 for xi,yi in zip(l.x, l.y)])/(len(l.x)-2))
    >>> round(l.s_y, 6) == round(s_y, 6)
    True
    """

    super(LinearCurve, self).__init__(x, y)
    self.slope, self.y_intercept, self.r_value, p_value, stderr = stats.linregress(self.x, self.y)

    # Compute standard deviation in residuals
    # see http://en.wikipedia.org/wiki/Calibration_curve
    self.s_y = math.sqrt(sum([(yi-self.slope*xi-self.y_intercept)**2 for xi,yi in zip(self.x, self.y)])/
                         (len(self.x)-2))


  @property
  def r_squared(self):
    """
    Returns the R^2 value of the curve. R is the correlation between the x and
    y values.

    >>> LinearCurve([1,2,3],[3,6,9]).r_squared
    1.0
    >>> x = np.random.random(10)
    >>> y = np.random.random(10)
    >>> round(LinearCurve(x, y).r_squared, 6) == round(stats.pearsonr(x, y)[0]**2, 6)
    True
    """
    return self.r_value**2

  def interpolate(self, y_unknown, replicates=1):
    """
    Find x value corresponding to the y value using the curve. Returns 

    (interpolated_x, err_f, min_x, max_x)

    interpolated_x: value interpolated from the linear curve. None if y_unknown
    is below or above limits of quantification.

    err_f: error function taking confidence (e.g. 0.99) as argument and
    returning error. You can interpret x with error as

      (interpolated_x-err_f(p), interpolated_x+err_f(p))

    min_x: if y_unknown is below limit of quantification, this is the lowest
    quantifiable x value that is higher than the x value that would have been
    extrapolated for y_unknown.

    max_x: if y_unknown is above limit of quantification, this is the highest
    quantifiable x value that is lower than the x value that would have been
    extraploated for y_unknown.

    >>> c = LinearCurve([1,2,3],[3,6,9])
    >>> x, err, min_x, max_x = c.interpolate(7.5)
    >>> x
    2.5
    >>> err(0.95)
    0.0

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12])
    >>> x, err, min_x, max_x = c.interpolate(7.5)
    >>> round(x, 2)
    2.54
    >>> round(err(0.95), 2)
    0.48

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12])
    >>> x, err, min_x, max_x = c.interpolate(7.5, replicates=3)
    >>> round(x, 2)
    2.54
    >>> round(err(0.95), 2)
    0.33

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12])
    >>> x, err, min_x, max_x = c.interpolate(12.1)
    >>> set([x, err, min_x]) == set([None])
    True
    >>> max_x
    4.0

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12])
    >>> x, err, min_x, max_x = c.interpolate(0.9)
    >>> set([x, err, max_x]) == set([None])
    True
    >>> min_x
    1.0

    """

    x_interpolated = (y_unknown-self.y_intercept)*1.0/self.slope

    # if x_interpolated is below smallest x, return smallest x as min_x
    if x_interpolated < min(self.x):
      return None, None, min(self.x)*1.0, None

    # if x_interpolated is above highest x, return highest x as max_x
    if x_interpolated > max(self.x):
      return None, None, None, max(self.x)*1.0

    # see http://en.wikipedia.org/wiki/Calibration_curve
    xm = np.mean(self.x)
    s_x = (self.s_y*1.0/abs(self.slope)) * math.sqrt(
      1.0/replicates +
      1.0/len(self.x) + 
      (y_unknown - np.mean(self.y))**2 / (self.slope**2 * sum([(xi-xm)**2 for xi in self.x]))
    )

    # two tailed error mode: take in confidence as a fraction, e.g. 0.99.
    # divide 2 for two-tailed test since stats.t.ppf is one-tailed. degree of
    # freedom is n-2.
    #
    # see
    # http://stackoverflow.com/questions/19339305/python-function-to-get-the-t-statistic
    # http://www.chem.utoronto.ca/coursenotes/analsci/StatsTutorial/ConcCalib.html

    err = lambda conf_frac: stats.t.ppf(1-(1-conf_frac)*0.5, len(self.x)-2)*s_x
    return x_interpolated, err, None, None


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
    [(3, 2), (3, 2), (4, 4), (5, 6)]

    >>> l = CalibrationCurve([1,2,3,4],[2,2,2,2]).linear_region()
    >>> l is None
    True

    When computing linear region, it averages y values with the same x value.

    >>> x = (2.7, 2.7, 2.7, 2.7, 8.6, 8.6, 8.6, 50.0, 50.0, 50.0, 50.0, 50.0)
    >>> y = (931451, 990078, 1269656, 1191860, 4491733, 181446, 4385907, 5863033, 6025161, 9218105, 8091820, 340011)
    >>> l = CalibrationCurve(x, y).linear_region()
    >>> l.x == x
    True
    >>> l.y == y
    True
    """

    if self.__linear_region is not None:
      return self.__linear_region

    # group data pts
    grouped_x = []
    grouped_y = []
    for i, p in enumerate(zip(self.x, self.y)):
      x = p[0]
      y = p[1]
      if len(grouped_x) == 0 or grouped_x[-1] != x:
        grouped_x.append(x)
        grouped_y.append([])
      grouped_y[-1].append(y)
    grouped_y = [np.mean(y) for y in grouped_y]

    ELBOW_R = 0.6
    linear_pos = []

    # find first elbow
    for i in range(2, len(grouped_x)):
      x = grouped_x[0:i]
      y = grouped_y[0:i]
      r = stats.pearsonr(x,y)[0]
      if r is not None and r >= ELBOW_R:
        linear_pos.append(i-2)
        break

    if len(linear_pos) == 0: # cannot find start of linear region, or just bad data!
      return None

    # find second elbow
    for i in range(2, len(grouped_x)-linear_pos[0]):
      x = grouped_x[len(grouped_x)-i:len(grouped_x)]
      y = grouped_y[len(grouped_x)-i:len(grouped_x)]
      r = stats.pearsonr(x,y)[0]
      if r is not None and r >= ELBOW_R:
        linear_pos.append(len(grouped_x)-i+1)
        break

    if len(linear_pos) == 1: # all linear
      linear_pos.append(len(grouped_x)-1)

    # linear_pos are indices in grouped_x, convert to indices for self.x
    linear_x0 = grouped_x[linear_pos[0]]
    linear_x1 = grouped_x[linear_pos[1]]

    linear_pos = [None, None]
    for i, x in enumerate(self.x):
      if x == linear_x0 and linear_pos[0] is None: # get index of first linear_x0
        linear_pos[0] = i
      if x == linear_x1: # get index of last linear_x1
        linear_pos[1] = i

    self.__lol = tuple(linear_pos)
    self.__linear_region = LinearCurve(self.x[linear_pos[0]:linear_pos[1]+1],
                                       self.y[linear_pos[0]:linear_pos[1]+1])
    return self.__linear_region

  def lol(self):
    """
    Returns limits of linearity.

    >>> CalibrationCurve([1,2,3],[2,4,6]).lol()
    (1, 3)

    >>> CalibrationCurve([1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]).lol()
    (1, 8)

    >>> CalibrationCurve([1,1,3,3,4,5,6,7,8],[2,2,2,2,4,6,10,8,8]).lol()
    (3, 5)

    >>> CalibrationCurve([1,2,3,4],[2,2,2,2]).lol() is None
    True
    """

    if self.__lol is None:
      self.linear_region()
    if self.__lol is not None:
      return (self.x[self.__lol[0]], self.x[self.__lol[1]])
    return None


if __name__ == '__main__':
  import doctest
  doctest.testmod()
