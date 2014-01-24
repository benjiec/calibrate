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

    >>> c = LinearCurve([1, 5],[10, 50])
    >>> x, err, min_x, max_x = c.interpolate(40)
    >>> x
    4.0
    >>> err(0.95)
    nan

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


if __name__ == '__main__':
  import doctest
  doctest.testmod()
