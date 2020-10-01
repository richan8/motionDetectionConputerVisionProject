### MODULES USED: OPENCV (cv2), NUMPY
import cv2
import numpy as np
from matplotlib import pyplot as plt

def normalize(img):
  return(255.0 * ((img - np.min(img)) * (1/(np.max(img) - np.min(img)))))

def spatialPartialDerivatives(img, save = False):
  imgIx = np.zeros((img.shape))
  imgIy = np.zeros((img.shape))
  for i in range(len(img)):
    for j in range(len(img[0])):
      if(i > 0 and j > 0):
        imgIx[i,j] = int(img[i, j]) - int(img[i, j-1])
        imgIy[i,j] = int(img[i, j]) - int(img[i-1, j])

  if(save):
    cv2.imwrite('Ix.png', normalize(imgIx))
    cv2.imwrite('Iy.png', normalize(imgIy))

  return(imgIx, imgIy)

def temporalPartialDerivate(img1, img2, save = False):
  imgIt = np.zeros((img1.shape))
  for i in range(len(img1)):
    for j in range(len(img1[0])):
      imgIt[i,j] = int(img2[i, j]) - int(img1[i, j])

  if(save):
    cv2.imwrite('It.png', normalize(imgIt))
  return(imgIt)

def opticalFlow(img,u,v, subsample = 5, save = False, nameFormat = ''):
  mag = np.sqrt(u**2 + v ** 2)
  print('minmax ',np.min(mag),np.max(mag))
  for i in range(mag.shape[0]):
    for j in range(mag.shape[1]):
      if(mag[i,j] < 0.3):
        mag[i, j] = 0
        u[i, j] = 0
        v[i, j] = 0
  
  if(save):
    cv2.imwrite(nameFormat + '_OF_Mag.png',normalize(mag))
    #FROM HINTS PROVIDED
    sub_u = u[0:img.shape[0]:subsample, 0:img.shape[1]:subsample]
    sub_v = v[0:img.shape[0]:subsample, 0:img.shape[1]:subsample]
    xc = np.linspace(0, img.shape[1], sub_u.shape[1])
    yc = np.linspace(0, img.shape[0], sub_u.shape[0])
    xv, yv = np.meshgrid(xc, yc)
    fig1 = plt.figure(figsize=(14, 7))
    plt.imshow(img, cmap='gray')
    plt.quiver(xv, yv, sub_u, sub_v, color='y')
    plt.savefig(nameFormat + '_OF.png')
    plt.clf()
  return(mag)

def lucasKanade(img1, img2, n, save = False, nameFormat = '', subsample = 5):
  u = np.zeros(img1.shape)
  v = np.zeros(img1.shape)
  imgIx, imgIy = spatialPartialDerivatives(img1, save = save)
  imgIt = temporalPartialDerivate(img1, img2, save = save)

  for i in range(n, img1.shape[0] - (int(n/2)+1)):
    for j in range(n, img1.shape[1] - (int(n/2)+1)):
      AIx = imgIx[i - int(n/2): i + 1 + int(n/2), j - int(n/2): j + 1 + int(n/2)].reshape((n*n, 1))
      AIy = imgIy[i - int(n/2): i + 1 + int(n/2), j - int(n/2): j + 1 + int(n/2)].reshape((n*n, 1))
      A = np.concatenate((AIx, AIy), axis=1)
      B = -1 * imgIt[i - int(n/2): i + 1 + int(n/2), j - int(n/2): j + 1 + int(n/2)].reshape((n*n, 1))
      X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
      u[i, j] = X[0, 0]
      v[i, j] = X[1, 0]

  opticalFlow(img1,u,v, save = save, nameFormat = nameFormat, subsample = subsample)
  return()

def computeCost(u, v, imgIx, imgIy, imgIt, L):
  cost = 0
  for i in range(1, u.shape[0]-1):
    for j in range(1, u.shape[1]-1):
      Es = (1/4)*((u[i, j] - u[i+1, j])**2 + (u[i, j] - u[i, j+1])**2 + (v[i, j] - v[i+1, j])**2 + (v[i, j] - v[i, j+1])**2)
      Ed = (imgIx[i, j]*u[i,j] + imgIy[i, j]*v[i, j] + imgIt[i, j])**2
      cost += Ed + L*Es
  return(cost)

def hornSchunck(img1, img2, minDiff = 0.01, maxItr = 600, L = 1, save = False, nameFormat = '', subsample = 5):
  u = np.zeros(img1.shape)
  v = np.zeros(img1.shape)
  imgIx, imgIy = spatialPartialDerivatives(img1, save = save)
  imgIt = temporalPartialDerivate(img1, img2, save = save)
  
  imgIx = imgIx
  imgIy = imgIy
  imgIt = imgIt
  
  print('minmax: %f - %f'%(np.min(imgIx), np.max(imgIy)))

  converged = False
  itr = 0
  cost = None

  while(not converged):
    for i in range(1, img1.shape[0]-1):
      for j in range(1, img1.shape[1]-1):
        uBar = (1/4)*(u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
        vBar = (1/4)*(v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1])
        num = imgIx[i, j]*uBar + imgIy[i, j]*vBar + imgIt[i, j]
        den = L**2 + imgIx[i, j]**2 + imgIy[i, j]**2
        u[i,j] = uBar - (imgIx[i, j])*(num/den)
        v[i,j] = vBar - (imgIy[i, j])*(num/den)
    
    newCost = computeCost(u, v, imgIx, imgIy, imgIt, L)

    if((cost is not None) and (abs(newCost - cost) < minDiff or (newCost > cost)) or (itr >= maxItr)):
      print('CONVERGED')
      converged = True

    if(cost is not None):
      deltaCost = newCost - cost

    deltaCost = ' - '
    if(cost):
      deltaCost = newCost - cost

    cost = newCost
    itr += 1
    print('ITR: #%i COST: %f DELTA COST: %s'%(itr, newCost, deltaCost))
  
  opticalFlow(img1, u, v, save = save, nameFormat = nameFormat, subsample = subsample)
  return()

def processImg(img):
  #plt.imshow(img, cmap='gray')
  #plt.show()
  img = cv2.medianBlur(np.float32(img), 3)
  #plt.imshow(img, cmap='gray')
  #plt.show()
  return(img)

#READING IMAGE
img1 = normalize(np.array(cv2.imread('traffic0.png', cv2.IMREAD_GRAYSCALE)))
img2 = normalize(np.array(cv2.imread('traffic1.png', cv2.IMREAD_GRAYSCALE)))

img1 = processImg(img1)
img2 = processImg(img2)

#nList = [3, 5, 11, 21]
#lList = [100, 10, 1, 0.1]
#for n in nList:
#  lucasKanade(img1, img2, n , save = True, nameFormat = 'LK_'+str(n))
#for L in lList: 
  #hornSchunck(img1, img2, L = L, save = True, nameFormat = 'HS_'+str(L))
#n = 15
#lucasKanade(img1, img2, n , save = True, nameFormat = 'LK2_'+str(n), subsample = 13)

L = 80
hornSchunck(img1, img2, L = L, save = True, nameFormat = 'HS_'+str(L), subsample = 13)


img1 = processImg(img1)