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

  def __init__(self, x, y, min_x=None, max_x=None):
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
    self.slope, self.y_intercept, self.r_value, self.p_value, self.stderr = stats.linregress(self.x, self.y)

    # Compute standard deviation in residuals
    # see http://en.wikipedia.org/wiki/Calibration_curve
    self.s_y = math.sqrt(sum([(yi-self.slope*xi-self.y_intercept)**2 for xi,yi in zip(self.x, self.y)])/
                         (len(self.x)-2))

    # Limits of linearity
    if min_x is None or min_x < min(self.x):
      min_x = min(self.x)
    if max_x is None or max_x > max(self.x):
      max_x = max(self.x)
    self.min_x = min_x
    self.max_x = max_x

  def fit_points(self):
    """
    Returns fitted curve as x-y points.

    >>> LinearCurve([1,2,3],[3,6,9]).fit_points()
    ((1.0, 3.0), (2.0, 6.0), (3.0, 9.0))
    >>> LinearCurve([1,2],[3,6]).fit_points()
    ((1.0, 3.0), (2.0, 6.0))
    >>> LinearCurve([1,2,3],[3,3,9]).fit_points()
    ((1.0, 2.0), (2.0, 5.0), (3.0, 8.0))
    """

    nvalues = max(2, len(self.x))
    bin = 1.0*(max(self.x)-min(self.x))/(nvalues-1)
    x_values = [min(self.x)+i*bin for i in range(0, nvalues)]
    return tuple([(x, self.slope*x+self.y_intercept) for x in x_values])

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

    (est_x, err_f, min_x, max_x)

    est_x: value interpolated or extrapolated from the linear curve.

    err_f: error function taking confidence (e.g. 0.99) as argument and
    returning error. You can interpret x with error as

      (interpolated_x-err_f(p), interpolated_x+err_f(p))

    min_x: if y_unknown is below limit of quantification and est_x is
    extrapolated, this is the lowest quantifiable x value that is higher than
    the x value that would have been extrapolated for y_unknown.

    max_x: if y_unknown is above limit of quantification and est_x is
    extrapolated, this is the highest quantifiable x value that is lower than
    the x value that would have been extraploated for y_unknown.

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
    >>> round(x,2)
    4.1
    >>> min_x
    1.0
    >>> max_x
    4.0

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12],1,4.2)
    >>> x, err, min_x, max_x = c.interpolate(12.1)
    >>> round(x,2)
    4.1
    >>> max_x
    4.0

    >>> c = LinearCurve([1,2,3,4],[3,6,8.5,12])
    >>> x, err, min_x, max_x = c.interpolate(0.9)
    >>> round(x, 2)
    0.31
    >>> min_x
    1.0
    >>> max_x
    4.0

    """

    x_est = (y_unknown-self.y_intercept)*1.0/self.slope

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
    return x_est, err, self.min_x*1.0, self.max_x*1.0


class PowerCurve(BaseCurve):

  def __init__(self, x, y, min_x=None, max_x=None):
    """
    Fit x and y through a power function y=mx^n

    >>> PowerCurve([0.01,0.1,1,10],[10,100,1000,10000]).scaling_factor
    1000.0
    >>> PowerCurve([0.01,0.1,1,10],[10,100,1000,10000]).exponent
    1.0
    >>> PowerCurve([0.01,0.1,1,10],[10,100,1000,10000]).s_y
    0.0
    >>> c = PowerCurve([0.01,0.1,1,10],[10,100,1000,9000])
    >>> r = math.sqrt(((10-(c.scaling_factor*(0.01**c.exponent)))**2+(100-(c.scaling_factor*(0.1**c.exponent)))**2+(1000-(c.scaling_factor*(1**c.exponent)))**2+(9000-(c.scaling_factor*(10**c.exponent)))**2)/(4-2))
    >>> c.s_y == r
    True
    """

    super(PowerCurve, self).__init__(x, y)

    # Limits of linearity
    if min_x is None or min_x < min(self.x):
      min_x = min(self.x)
    if max_x is None or max_x > max(self.x):
      max_x = max(self.x)
    self.min_x = min_x
    self.max_x = max_x

    if self.min_x <= 0:
      raise Exception('Cannot handle non-positive x values')

    log_x = [math.log10(n) for n in x]
    log_y = [math.log10(n) for n in y]
    self.log_linear = LinearCurve(log_x, log_y, math.log10(self.min_x), math.log10(self.max_x))
    self.scaling_factor = 10**(self.log_linear.y_intercept)
    self.exponent = self.log_linear.slope

    # Compute standard deviation in residuals
    self.s_y = math.sqrt(sum([(yi-self.scaling_factor*(xi**self.exponent))**2
                              for xi,yi in zip(self.x, self.y)])
                         /(len(self.x)-2))

  def fit_points(self):
    """
    Returns fitted curve as x-y points.

    >>> PowerCurve([0.01,0.1,1,10],[10,100,1000,10000]).fit_points()
    ((0.01, 10.0), (0.1, 100.0), (1.0, 1000.0), (10.0, 10000.0))
    """

    nvalues = max(2, len(self.x))
    bin = 1.0*(math.log10(max(self.x))-math.log10(min(self.x)))/(nvalues-1)
    log_x_values = [math.log10(min(self.x))+i*bin for i in range(0, nvalues)]
    x_values = [10**n for n in log_x_values]
    return tuple([(x, self.scaling_factor*(x**self.exponent)) for x in x_values])

  def interpolate(self, y_unknown, replicates=1):
    """
    Find x value corresponding to the y value using the curve. Returns 

    (est_x, err_f, min_x, max_x)

    See LinearCurve.interpolate for detail.

    >>> c = PowerCurve([0.01,0.1,1,10],[10,100,1000,10000])
    >>> x, err, min_x, max_x = c.interpolate(500)
    >>> round(x,1)
    0.5
    >>> err(0.95)
    0.0

    >>> c = PowerCurve([0.01,0.1,1,10],[10,100,900,9000])
    >>> x, err, min_x, max_x = c.interpolate(500)
    >>> round(x,2)
    0.53
    >>> round(err(0.95),4)
    0.0876

    >>> c = PowerCurve([0.01,0.1,1,10],[100,1000,6000,90000])
    >>> x, err, min_x, max_x = c.interpolate(530)
    >>> round(x,2)
    0.06
    >>> round(err(0.95),4)
    0.1096
    """

    log_est = self.log_linear.interpolate(math.log10(y_unknown), replicates=replicates)

    if log_est[0] is None:
      return [n if n is None else 10**n for n in log_est]

    log_err = log_est[1]

    def err(conf_frac):
      err = log_err(conf_frac)
      rng = (log_est[0]-err, log_est[0]+err)
      rng = [10**n for n in rng]
      return (rng[1]-rng[0])*1.0/2

    return 10**log_est[0], err, 10**log_est[2], 10**log_est[3]


if __name__ == '__main__':
  import doctest
  doctest.testmod()
