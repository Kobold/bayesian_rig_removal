#!/usr/bin/python2.6
from matplotlib import nxutils
from scipy import ndimage
from numpy import linalg
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import numpy as np

SIGMA_E = 0.01 # allowance of acceleration
SIGMA_V = 0.01 # allowance of acceleration
LAMBDA = 2.0
ALPHA = 0.0331 # ~ 3 * SIGMA_V according to section 3.4


def rig_matte((height, width), vectors):
    """
    height, width - dimensions of the output matrix
    vectors - a list of (x, y) tuple vectors that specify the vertices of a polygonal rig region
    """
    img = Image.new('L', (width, height), 1)
    ImageDraw.Draw(img).polygon(vectors, outline=0, fill=0)
    return np.array(img, dtype=np.float_)


def pl(d_prev, w_n, w_n_1, I_n, I_n_1):
    """p_l - Image data likelihood (equation 3)
    
    x_r
        a rig site
    w_n
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    w_n_1: w_n-1
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    I_n
        frame at n
    I_n_1: I_n-1
        frame at n-1
    """
    rows, cols = d_prev.shape[:2]
    w_compensated = np.zeros((rows, cols))
    I_compensated = np.zeros((rows, cols))
    
    for row in xrange(rows):
        for col in xrange(cols):
            y_motion, x_motion = d_prev[row, col]
            y_prime = round(row + y_motion) # the motion compensated site x_r + d^h_n,n-1(x_r)
            x_prime = round(col + x_motion)
            
            x_r = (row, col) # y, x
            x_r_prime = (min(max(0, y_prime), rows - 1),
                         min(max(0, x_prime), cols - 1))
            
            w_compensated[x_r] = w_n_1[x_r_prime]
            I_compensated[x_r] = I_n_1[x_r_prime]
    
    temp = (1. / (2. * (SIGMA_E ** 2.))) * w_n * w_compensated * \
           (I_n - I_compensated) ** 2.
    
    return np.exp(-temp)

def pt(x_r, x_r_prime, occlusion, w_n_1, d_h, d_prev):
    """p_t - Temporal smoothness (equation 4)
    
    x_r:
        a rig site
    occlusion: o_n,n-1
        binary matrix. 1 indicates data that that point in the frame n does not
        exist in the frame n-1. 0 indicates no discontinuity
    w_n_1: w_n-1
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    d_h: d^h_n,n-1
        vector matrix estimating the motion of the hidden area
    d_prev: d_n-1,n-2
        vector matrix with the motion mapping from frame n-1 to frame n-2
    """
    temp = (1. / SIGMA_V ** 2.) * \
           (1 - occlusion[x_r]) * \
           w_n_1[x_r_prime] * \
           linalg.norm(d_h[x_r] - d_prev[x_r_prime]) ** 2.
    
    return math.exp(-temp)

def ps(d_h):
    """p_s - Spatial motion smoothness (equation 5)
    
    Calculates it as a batch rather than per pixel.
    """
    def norm(vectors):
        return vectors[:,:,0] ** 2. + vectors[:,:,1] ** 2.
    
    z = np.zeros(d_h.shape[:2])
    t1, t2, t3, t4, t5, t6, t7, t8 = [z.copy() for x in range(8)]
    
    t1[1:,:]  = 2. * norm(d_h[1:,:]  - d_h[:-1,:])
    t3[:,:-1] = 2. * norm(d_h[:,:-1] - d_h[:,1:])
    t5[:-1,:] = 2. * norm(d_h[:-1,:] - d_h[1:,:])
    t7[:,1:]  = 2. * norm(d_h[:,1:]  - d_h[:,:-1])
    
    t2[1:,:-1]  = 1.4142135623730949 * norm(d_h[1:,:-1]  - d_h[:-1,1:])
    t4[:-1,:-1] = 1.4142135623730949 * norm(d_h[:-1,:-1] - d_h[1:,1:])
    t6[:-1,1:]  = 1.4142135623730949 * norm(d_h[:-1,1:]  - d_h[1:,:-1])
    t8[1:,1:]   = 1.4142135623730949 * norm(d_h[1:,1:]   - d_h[:-1,:-1])
    
    temp = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    
    return np.exp(-temp)
    

def pso(occlusion):
    """p_so - Spatial occlusion smoothness
    
    Calculates it as a batch rather than per pixel.
    """
    z = np.zeros(occlusion.shape)
    t1, t2, t3, t4, t5, t6, t7, t8 = [z.copy() for x in range(8)]
    
    t1[1:,:]  = 2. * abs(occlusion[1:,:]  - occlusion[:-1,:])
    t3[:,:-1] = 2. * abs(occlusion[:,:-1] - occlusion[:,1:])
    t5[:-1,:] = 2. * abs(occlusion[:-1,:] - occlusion[1:,:])
    t7[:,1:]  = 2. * abs(occlusion[:,1:]  - occlusion[:,:-1])
    
    t2[1:,:-1]  = 1.4142135623730949 * abs(occlusion[1:,:-1]  - occlusion[:-1,1:])
    t4[:-1,:-1] = 1.4142135623730949 * abs(occlusion[:-1,:-1] - occlusion[1:,1:])
    t6[:-1,1:]  = 1.4142135623730949 * abs(occlusion[:-1,1:]  - occlusion[1:,:-1])
    t8[1:,1:]   = 1.4142135623730949 * abs(occlusion[1:,1:]   - occlusion[:-1,:-1])
    
    temp = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
    
    penalty = ALPHA * occlusion.sum()
    
    return math.exp(-penalty) * np.exp(-temp)

def main(im1, im2, im3):
    I_n = im3
    I_n_1 = im2
    
    # ghetto build of occlusion
    height, width = im1.shape
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon([(679, 270), (719, 264), (742, 339), (680, 340)], outline=1, fill=0)
    occlusion = np.array(img, dtype=np.float_)
    
    d_prev_x = np.genfromtxt(file('d_prev_x.csv'), delimiter=',')
    d_prev_y = np.genfromtxt(file('d_prev_y.csv'), delimiter=',')
    
    assert I_n.shape == d_prev_x.shape
    
    # make a 3d height x width x 2 matrix to hold the vectors
    d_prev = np.zeros(list(d_prev_x.shape) + [2])
    d_prev[:, :, 0] = d_prev_y # note, this y here is correct--and it's important it be this order
    d_prev[:, :, 1] = d_prev_x
    
    # ghetto build of estimate hidden motion
    d_h = np.zeros(d_prev.shape)
    d_h[:, :, 0] = 0.0
    d_h[:, :, 1] = -1.0
    
    # w_n - weight field for frame 3
    # w_n_1 - weight field for frame 2
    w_n = rig_matte(im1.shape, [(679, 270), (719, 264), (742, 339), (680, 340)])
    w_n_1 = rig_matte(im1.shape, [(679, 273), (726, 263), (740, 334), (679, 337)])
    
    pl_matrix = pl(d_prev, w_n, w_n_1, I_n, I_n_1)
    ps_matrix = ps(d_h)
    pso_matrix = pso(occlusion)
    
    results = np.zeros(im1.shape)
    rows, cols = im1.shape
    for row in xrange(rows):
        for col in xrange(cols):
            y_motion, x_motion = d_prev[row, col]
            y_prime = round(row + y_motion) # the motion compensated site x_r + d^h_n,n-1(x_r)
            x_prime = round(col + x_motion)
            
            x_r = (row, col) # y, x
            x_r_prime = (min(max(0, y_prime), rows - 1),
                         min(max(0, x_prime), cols - 1))
            
            results[row, col] = pl_matrix[x_r] * \
                                pt(x_r, x_r_prime, occlusion, w_n_1, d_h, d_prev) * \
                                ps_matrix[x_r] * \
                                pso_matrix[x_r]
    
    return results
    
    
"""
%Code
for i = 1:rows
    for j = 1:cols
        x_r = [i, j];
        
        p = (pl(x_r, w_n, w_n_1, d_h, I_n, I_n_1).* ...
        pt(x_r, occlusion, w_n_1, d_h, d_prev)) * ...
        ps(x_r, d_prev, d_prev) * ...
        pso(x_r, occlusion);
        
        p = ((pl(x_r, w_n, w_n_1, d_prev, I_n, I_n_1).*pt(x_r, occlusion, w_n_1, d_prev, d_prev)) * ps(x_r, d_prev, d_prev) * pso(x_r, occlusion));
    end
end    

end
"""


im1 = ndimage.imread('Forest_Gump/001.png', flatten=True)
im2 = ndimage.imread('Forest_Gump/002.png', flatten=True)
im3 = ndimage.imread('Forest_Gump/003.png', flatten=True)

bob = main(im1, im2, im3)

#plt.imshow(bob / bob.max(), cmap='gray')
#plt.show()
#import pdb
#pdb.set_trace()












