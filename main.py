#!/usr/bin/python2.6
from matplotlib import nxutils
from scipy import misc, ndimage
from numpy import linalg
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
import numpy as np

SIGMA_E = 0.01 # allowance of acceleration
SIGMA_V = 0.01 # allowance of acceleration
LAMBDA = 2.0
ALPHA = 0.0331 # ~ 3 * SIGMA_V according to section 3.4
DOWNSAMPLING = 0

def downsample(arr, x=DOWNSAMPLING):
    """Returns m x n matrix ``a`` downsampled to a (m / 2^x) x (n / 2^x) matrix."""
    return misc.imresize(arr, 0.5 ** x, interp='bilinear', mode='F')

def rig_matte((height, width), vectors, dtype=np.float_):
    """
    height, width - dimensions of the output matrix
    vectors - a list of (x, y) tuple vectors that specify the vertices of a polygonal rig region
    """
    img = Image.new('L', (width, height), 1)
    ImageDraw.Draw(img).polygon(vectors, outline=0, fill=0)
    return np.array(img, dtype=dtype)


def pl(w_n, w_compensated, I_n, I_compensated):
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
    temp = (1. / (2. * (SIGMA_E ** 2.))) * w_n * w_compensated * \
           (I_n - I_compensated) ** 2.
    
    return np.exp(-temp)

def norm(vectors):
    return vectors[:,:,0] ** 2. + vectors[:,:,1] ** 2.

def pt(occlusion, w_compensated, d_h, d_prev_compensated):
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
    temp = (1. / SIGMA_V ** 2.) * (1. - occlusion) * w_compensated * \
           norm(d_h - d_prev_compensated)
    
    return np.exp(-temp)

def ps(d_h):
    """p_s - Spatial motion smoothness (equation 5)
    
    Calculates it as a batch rather than per pixel.
    """
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
    
    d_prev = load_d('vel_002_001')
    assert I_n.shape == d_prev.shape[:2]
    
    # ghetto build of estimate hidden motion
    d_h = np.zeros(d_prev.shape)
    d_h[:, :, 0] = 0.0
    d_h[:, :, 1] = -1.0
    
    # w_n - weight field for frame 3
    # w_n_1 - weight field for frame 2
    w_n = rig_matte(im1.shape, [(679, 270), (719, 264), (742, 339), (680, 340)])
    w_n_1 = rig_matte(im1.shape, [(679, 273), (726, 263), (740, 334), (679, 337)])
    
    # compute versions of various matrices compensated by d_prev.
    # this is expensive so we make sure to only do this loop once
    d_prev_compensated = np.zeros(d_prev.shape)
    I_compensated = np.zeros(im1.shape)
    w_compensated = np.zeros(im1.shape)
    
    rows, cols = im1.shape
    for row in xrange(rows):
        for col in xrange(cols):
            y_motion, x_motion = d_prev[row, col]
            y_prime = round(row + y_motion) # the motion compensated site x_r + d^h_n,n-1(x_r)
            x_prime = round(col + x_motion)
            
            x_r = (row, col) # y, x
            x_r_prime = (min(max(0, y_prime), rows - 1),
                         min(max(0, x_prime), cols - 1))
            
            d_prev_compensated[x_r] = d_prev[x_r_prime]
            I_compensated[x_r] = I_n_1[x_r_prime]
            w_compensated[x_r] = w_n_1[x_r_prime]
    
    # compute the probability equations
    pl_matrix = pl(w_n, w_compensated, I_n, I_compensated)
    pt_matrix = pt(occlusion, w_compensated, d_h, d_prev_compensated)
    ps_matrix = ps(d_h)
    pso_matrix = pso(occlusion)
    
    return pl_matrix * pt_matrix * ps_matrix * pso_matrix

def load_d(prefix):
    """Loads the displacement matrix from csv files. Returns a height x width x 2 matrix."""
    vel_x = downsample(np.genfromtxt(file('%s_x.csv' % prefix), delimiter=','))
    vel_y = downsample(np.genfromtxt(file('%s_y.csv' % prefix), delimiter=','))

    # make a 3d height x width x 2 matrix to hold the vectors
    vel = np.zeros(list(vel_x.shape) + [2])
    vel[:, :, 0] = vel_y # note, this y here is correct--and it's important it be this order
    vel[:, :, 1] = vel_x
    return vel

def vector_weighted_average(vf, weights):
    """Returns the average vector of vector field ``vf``."""
    weights_sum = weights.sum()
    y_average = (vf[:,:,0] * weights).sum() / weights_sum
    x_average = (vf[:,:,1] * weights).sum() / weights_sum
    return np.array([y_average, x_average])

def bounding_box(vertices, (height, width), extend=5):
    """Returns the bounding box of ``vertices`` plus some boundary ``extend``.
    
    Returned bounding box has format (x_min, x_max, y_min, y_max)
    """
    x_min = min(x for x, y in vertices) - extend
    x_max = max(x for x, y in vertices) + extend
    y_min = min(y for x, y in vertices) - extend
    y_max = max(y for x, y in vertices) + extend
    
    return max(x_min, 0), min(x_max, width), max(y_min, 0), min(y_max, height)

def spatial_interpolation_vector(d, rig_vertices):
    """Returns the vector for all the spatially interpolated candidates. (section 4.2)
    
    Note that this uses the second, ghetto, method of assuming everything behind the rig moves with
    one motion.
    """
    shape = d.shape[:2]
    x_min, x_max, y_min, y_max = bounding_box(rig_vertices, shape)
    matte = rig_matte(shape, vertices)
    return vector_weighted_average(d[y_min:y_max, x_min:x_max],
                                   matte[y_min:y_max, x_min:x_max])

def temporal_interpolation_vectors(d_prev, candidates, bounds):
    assert d_prev.shape[:2] == candidates.shape
    
    rows, cols = candidates.shape
    for index in index_iterator(bounds):
        c_row, c_col = np.array(index) - d_prev[index]
        if 0 <= c_row < rows and 0 <= c_col < cols:
            candidates[c_row, c_col].append(d_prev[index])

#
# Energy calculations
#

def E_l(x_r, x_r_prime, w_n, w_n_1, I_n, I_n_1):
    """E_l - Image data likelihood (equation 9)
    
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
    temp = (1. / (2. * SIGMA_E ** 2.)) * \
           w_n[x_r] * \
           w_n_1[x_r_prime] * \
           ((I_n[x_r] - I_n_1[x_r_prime]) ** 2.)
    
    return temp

def E_0_t(x_r, x_r_prime, w_n_1, candidate, d_prev):
    """E_0_t - Temporal smoothness (equation 9)

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
           w_n_1[x_r_prime] * \
           linalg.norm(candidate - d_prev[x_r_prime]) ** 2.

    return temp

def neighborhood((y, x), (height, width)):
    return [(yt, xt) for xt in [x + 1, x, x - 1]
                     for yt in [y + 1, y, y - 1]
                     if 0 <= xt < width and 0 <= yt < height
                     and (xt, yt) != (x, y)]

def lambda_(s, x_r):
    return LAMBDA / linalg.norm(np.array(s) - np.array(x_r))

def E_s(x_r, candidate, d_h):
    """E_s - Spatial motion smoothness (equation 9)"""
    temp = sum(lambda_(s, x_r) * linalg.norm(candidate - d_h[s]) ** 2.
               for s in neighborhood(x_r, d_h.shape[:2]))

    return temp

def E_0_o(x_r, occlusion):
    """E_0_o - Spatial occlusion smoothness"""
    temp = sum(lambda_(s, x_r) * abs(occlusion[s]) ** 2.
               for s in neighborhood(x_r, occlusion.shape[:2]))

    return temp

def E_1_o(x_r, occlusion):
    """E_1_o - Spatial occlusion smoothness"""
    temp = sum(lambda_(s, x_r) * abs(1. - occlusion[s]) ** 2.
               for s in neighborhood(x_r, occlusion.shape[:2]))

    return temp


#
# Tools
#

def vector_display(vf):
    """Draws the vector field magnitudes of ``vf``."""
    vf_mag = np.sqrt(norm(vf))
    plt.imshow(vf_mag / vf_mag.max(), cmap='gray', interpolation='nearest')
    plt.show()

def index_iterator((x_min, x_max, y_min, y_max)):
    for row in xrange(y_min, y_max):
        for col in xrange(x_min, x_max):
            yield (row, col)

if __name__ == '__main__':
    displacement = load_d('vel_003_002')
    shape = displacement.shape[:2]
    
    # calculate spatial interpolation vector (section 4.2)
    vertices = [(679, 270), (719, 264), (742, 339), (680, 340)]
    siv = spatial_interpolation_vector(displacement, vertices)
    
    # initialize the candidates for the motion with the spatial interpolation
    print 'initializing candidates'
    candidates = np.empty(shape, dtype=object)
    for index, y in np.ndenumerate(candidates):
        candidates[index] = [siv]
    
    # find temporal interpolation candidates (section 4.3)
    print 'candidate # =', sum(len(x) for x in candidates.flat)
    d_prev = load_d('vel_002_001')
    bounds = bounding_box(vertices, shape)
    temporal_interpolation_vectors(d_prev, candidates, bounds)
    print 'candidate # =', sum(len(x) for x in candidates.flat)
    
    # add adjacent neighbors as candidates if they've been assigned
    # w_n - weight field for frame 3
    # w_n_1 - weight field for frame 2
    print 'adding additional candidates'
    w_n = rig_matte(shape, [(679, 270), (719, 264), (742, 339), (680, 340)])
    w_n_1 = rig_matte(shape, [(679, 273), (726, 263), (740, 334), (679, 337)])
    for x_r in index_iterator(bounds):
        if w_n[x_r] == 1:
            candidate = displacement[x_r]
            for s in neighborhood(x_r, shape):
                if w_n[s] < 1:
                    candidates[s].append(candidate)
    print 'finished adding additional candidates'
    print 'candidate # =', sum(len(x) for x in candidates.flat)
    
    # candidate evaluation (section 4.4)
    occluded = np.logical_not(rig_matte(shape, vertices, dtype=bool))
    new_occluded = occluded.copy()
    perturb = np.random.randn(*d_prev.shape) / 4.
    d_h = np.where(np.dstack((occluded, occluded)),
                   np.tile(siv, list(shape) + [1]) + perturb, # initialize with spatial interp
                   displacement)
    new_d_h = d_h.copy()
    
    load = lambda fname: ndimage.imread(fname, flatten=True) / 255.
    im2 = load('Forest_Gump/002.png')
    im3 = load('Forest_Gump/003.png')
    
    is_rig = occluded.copy()
    while True:
        for x_r in index_iterator(bounds):
            if is_rig[x_r]:
                minimum_energy = float('inf')
                best_candidate = None # the candidate associated with the minimum energy
                best_is_occluded = None
            
                for candidate in candidates[x_r]:
                    # the motion compensated site x_r' = x_r + d^h_n,n-1(x_r)
                    x_r_prime = tuple((np.array(x_r) + candidate).round())
                
                    el = E_l(x_r, x_r_prime, w_n, w_n_1, im3, im2)
                    e0t = E_0_t(x_r, x_r_prime, w_n_1, candidate, d_prev)
                    e1t = ALPHA
                    es = E_s(x_r, candidate, d_h)
                    e0o = E_0_o(x_r, occluded)
                    e1o = E_1_o(x_r, occluded)
                
                    e0 = el + e0t + es + e0o
                    e1 = el + e1t + es + e1o
                
                    if e0 < minimum_energy:
                        minimum_energy = e0
                        best_candidate = candidate
                        best_is_occluded = False
                    if e1 < minimum_energy:
                        minimum_energy = e1
                        best_candidate = candidate
                        best_is_occluded = True
                    
                    #if e0 < e1:
                    #    print x_r, d_h[x_r], '->', candidate, min(e0, e1)
                    #    print '\t', 'e0 = %f + %f + %f + %f = %f' % (el, e0t, es, e0o, e0)
                    #    print '\t', 'e1 = %f + %f + %f + %f = %f' % (el, e1t, es, e1o, e1)
            
                if best_candidate is not None:
                    new_d_h[x_r] = best_candidate
                    new_occluded[x_r] = best_is_occluded
        
        print 'occluded changed:', (occluded != new_occluded).sum()
        print 'd_h changed:', (d_h != new_d_h).sum()
        if (occluded == new_occluded).all() and (d_h == new_d_h).all():
            break
        
        print 'iterating again'
        occluded = new_occluded
        d_h = new_d_h
    
    # reconstruct that shizzle
    I_h = im3.copy()
    for x_r in index_iterator(bounds):
        if is_rig[x_r]:
            x_r_prime = tuple((np.array(x_r) + d_h[x_r]).round())
            I_h[x_r] = (w_n[x_r] * im3[x_r] + w_n_1[x_r_prime] * im2[x_r_prime]) / (w_n[x_r] + w_n_1[x_r_prime])
    
    #load = lambda fname: downsample(ndimage.imread(fname, flatten=True))
    #im1 = load('Forest_Gump/001.png')
    #im2 = load('Forest_Gump/002.png')
    #im3 = load('Forest_Gump/003.png')
    #bob = main(im1, im2, im3)

    #plt.imshow(bob / bob.max(), cmap='gray')
    #plt.show()
    #import pdb
    #pdb.set_trace()
    
    a = np.array(range(3*3)).reshape((3, 3))
    
    
