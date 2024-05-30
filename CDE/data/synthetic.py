import numpy as np
import math
import scipy.stats as stats


class Bimodal():
  def __init__(self):
    self.sig = 0.05

  def TrueFunc(self, X):
    p = 1/(1+np.exp(-1.5*X))
    Y=np.zeros((len(X),1))
    for i, a in enumerate(p):
        Y[i,0] = np.random.binomial(1,a,1)
    Z = 0.2*X + Y + np.random.randn(len(Y),1)*(X*0.05)

    return Z

  def generate_train(self, N_x, N_y=1):
    X = np.random.uniform(-5, 5, (N_x,1))
    y = self.TrueFunc(X)

    return [X,y]

  def generate_test(self, N_y, N_x=200):
    Xp = np.array([np.linspace(-5,5,N_x)]).T
    Xp=np.repeat(Xp, N_y, axis=0)
    yp = self.TrueFunc(Xp)
    yp=np.reshape(yp, [N_x, N_y, 1])
    Xp=np.reshape(Xp, [N_x, N_y, 1])

    return [Xp,yp]
  

class Skewed():
  def __init__(self):
    self.loc_slope = 0.1
    self.loc_intercept = 0.0

    self.scale_square_param = 0.1
    self.scale_intercept = 0.05

    self.skew_low = -8
    self.skew_high = 0.0

  def _loc_scale_skew_mapping(self, X):
    loc = self.loc_intercept + self.loc_slope * X
    scale = self.scale_intercept + self.scale_square_param * np.abs(X)
    skew = self.skew_low + (self.skew_high - self.skew_low) * (1 / (1+np.exp(-X)))
    return loc, scale, skew

  def TrueFunc(self, X):
    locs, scales, skews = self._loc_scale_skew_mapping(X)
    rvs = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      rvs[i] = stats.skewnorm.rvs(skews[i], loc=locs[i], scale=scales[i])
    rvs = np.expand_dims(rvs, 1)

    return rvs

  def generate_train(self, N_x, N_y=1):
    X = np.random.uniform(-5, 5, (N_x,1))
    y = self.TrueFunc(X)

    return [X,y]

  def generate_test(self, N_y, N_x=200):
    Xp = np.array([np.linspace(-5,5,N_x)]).T
    Xp=np.repeat(Xp, N_y, axis=0)
    yp = self.TrueFunc(Xp)
    yp=np.reshape(yp, [N_x, N_y, 1])
    Xp=np.reshape(Xp, [N_x, N_y, 1])

    return [Xp,yp]
  
class Ring():
  def __init__(self):
    self.sig=0.1

  def TrueFunc(self, X):
    Y=np.zeros((len(X),1))
    for i, x in enumerate(X):
        coin1 = np.random.binomial(1, 0.5 ,1)
        if coin1 == 1 and np.abs(x)<=1:
          Y[i,0]=2*(np.random.rand(1, 1) - 0.5) * 1
        else:
          angle=np.degrees(np.arccos(x/2))
          coin2 = np.random.binomial(1, 0.5 ,1)
          if coin2 == 1:
            angle=angle
            Y[i,0]=2*np.sin(np.radians(angle))+np.random.randn(1) * self.sig
          else:
            angle=-angle+360
            Y[i,0]=2*np.sin(np.radians(angle))+np.random.randn(1) * self.sig
    return Y

  def generate_train(self, N_x, N_y=1):
    X = np.random.uniform(-2, 2, (N_x,1))
    y = self.TrueFunc(X)

    return [X,y]

  def generate_test(self, N_y, N_x=200):
    Xp = np.array([np.linspace(-2,2,N_x)]).T
    Xp=np.repeat(Xp, N_y, axis=0)
    yp = self.TrueFunc(Xp)
    yp=np.reshape(yp, [N_x, N_y, 1])
    Xp=np.reshape(Xp, [N_x, N_y, 1])

    return [Xp,yp]


def generate_bimodal(num_seeds: int, N_x: int, N_y: int):
    vX=np.zeros((num_seeds, N_x, 1))
    vy=np.zeros((num_seeds, N_x, 1))
    vXp = np.zeros((num_seeds, 200, N_y, 1))
    vyp = np.zeros((num_seeds, 200, N_y, 1))

    for i in range(num_seeds):
      np.random.seed(i)
      X, y=Bimodal().generate_train(N_x)
      Xp, yp=Bimodal().generate_test(N_y)

      vX[i] = X
      vy[i] = y
      vXp[i] = Xp
      vyp[i] = yp

    return vX, vy, vXp[:,:,0,:], vyp

def generate_skewed(num_seeds: int, N_x: int, N_y: int):
    vX=np.zeros((num_seeds, N_x, 1))
    vy=np.zeros((num_seeds, N_x, 1))
    vXp = np.zeros((num_seeds, 200, N_y, 1))
    vyp = np.zeros((num_seeds, 200, N_y, 1))
    
    for i in range(num_seeds):
      np.random.seed(i)
      X, y=Skewed().generate_train(N_x)
      Xp, yp=Skewed().generate_test(N_y)

      vX[i] = X
      vy[i] = y
      vXp[i] = Xp
      vyp[i] = yp

    return vX, vy, vXp[:,:,0,:], vyp

def generate_ring(num_seeds: int, N_x: int, N_y: int):
    vX=np.zeros((num_seeds, N_x, 1))
    vy=np.zeros((num_seeds, N_x, 1))
    vXp = np.zeros((num_seeds, 200, N_y, 1))
    vyp = np.zeros((num_seeds, 200, N_y, 1))
    
    for i in range(num_seeds):
      np.random.seed(i)
      X, y=Ring().generate_train(N_x)
      Xp, yp=Ring().generate_test(N_y)

      vX[i] = X
      vy[i] = y
      vXp[i] = Xp
      vyp[i] = yp

    return vX, vy, vXp[:,:,0,:], vyp