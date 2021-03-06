#!/usr/bin/python2.6
from scipy import ndimage
from numpy import linalg
from PIL import Image, ImageDraw
import numpy as np
import os
import re
import time

SIGMA_E = 0.01 # allowance of acceleration
SIGMA_V = 0.01 # allowance of acceleration
LAMBDA = 2.0
ALPHA = 0.0331 # ~ 3 * SIGMA_V according to section 3.4


#
# Helpers
#

def bounding_box(vertices, (height, width), extend=5):
    """Returns the bounding box of ``vertices`` plus some boundary ``extend``.
    
    Returned bounding box has format (x_min, x_max, y_min, y_max)
    """
    x_min = min(x for x, y in vertices) - extend
    x_max = max(x for x, y in vertices) + extend
    y_min = min(y for x, y in vertices) - extend
    y_max = max(y for x, y in vertices) + extend
    
    return max(x_min, 0), min(x_max, width), max(y_min, 0), min(y_max, height)

def frame_string(path):
    """Extracts a frame string like '004' from a path like 'Forest_Gump/004.png'."""
    filename = os.path.split(path)[1]
    return os.path.splitext(filename)[0]

def index_iterator((x_min, x_max, y_min, y_max)):
    """Yieds indices within a given range. Makes it easy to iterate over subsections of
    a matrix."""
    for row in xrange(y_min, y_max):
        for col in xrange(x_min, x_max):
            yield (row, col)

def load_d(prefix):
    """Loads optical flow displacement from csv. Returns a height x width x 2 matrix.
    
    prefix - the common prefix of the x and y optical flow files
    """
    vel_x = np.genfromtxt(file('%s_x.csv' % prefix), delimiter=',')
    vel_y = np.genfromtxt(file('%s_y.csv' % prefix), delimiter=',')

    # make a 3d height x width x 2 matrix to hold the vectors
    vel = np.zeros(list(vel_x.shape) + [2])
    vel[:, :, 0] = vel_y # note, this y here is correct--and it's important it be this order
    vel[:, :, 1] = vel_x
    return vel

def parse_rig_vertices(f):
    """Given the file ``f``, parses the vertices of the rig for each frame out of it."""
    vertices_list = []
    for line in f:
        vertices = line.split()
        point = [tuple(map(int, str_point.split(','))) for str_point in vertices]
        vertices_list.append(point)

    return vertices_list

def rig_matte((height, width), vectors, dtype=np.float_):
    """Returns a matte with a polygonal hole punched out.
    
    height, width - dimensions of the output matrix
    vectors - a list of (x, y) tuple vectors that specify the vertices of a polygonal rig region
    dtype - the type of matrix to return
    """
    img = Image.new('L', (width, height), 1)
    ImageDraw.Draw(img).polygon(vectors, outline=0, fill=0)
    return np.array(img, dtype=dtype)

def save_image(filename, matrix):
    """Saves ``matrix`` to a black and white image named ``filename``."""
    im = Image.new('L', list(reversed(matrix.shape)))
    data = np.floor(np.ravel(matrix) * 256)
    im.putdata(data)
    im.save(filename)

def vector_weighted_average(vf, weights):
    """Returns the average vector of vector field ``vf`` weighted by ``weights``."""
    weights_sum = weights.sum()
    y_average = (vf[:,:,0] * weights).sum() / weights_sum
    x_average = (vf[:,:,1] * weights).sum() / weights_sum
    return np.array([y_average, x_average])

#
# Interpolation for Candidate Motion Vectors
#

def spatial_interpolation_vector(d, rig_vertices):
    """Returns the vector for all the spatially interpolated candidates. (section 4.2)
    
    Note that this uses the second, more simple, method of assuming everything behind the rig
    moves with one motion.
    
    d - a displacement matrix
    rig_vertices - the vertices of an area to not include in interpolation
    """
    shape = d.shape[:2]
    x_min, x_max, y_min, y_max = bounding_box(rig_vertices, shape)
    matte = rig_matte(shape, rig_vertices)
    return vector_weighted_average(d[y_min:y_max, x_min:x_max],
                                   matte[y_min:y_max, x_min:x_max])

def temporal_interpolation_vectors(d_prev, candidates, bounds):
    """Interpolates motion candidates from previous frames. (section 4.3)
    
    d_prev - d_n-1,n-2 from the paper. Motion from frame n-1 to n-2.
    candidates - a matrix of lists--interpolated candidates will be added to the appropriate location
    bounds - a bounding box surrounding the rig area
    """
    assert d_prev.shape[:2] == candidates.shape
    
    rows, cols = candidates.shape
    for index in index_iterator(bounds):
        c_row, c_col = np.array(index) - d_prev[index]
        if 0 <= c_row < rows and 0 <= c_col < cols:
            candidates[c_row, c_col].append(d_prev[index])

#
# Motion energy components
#

def E_l(x_r, x_r_prime, w_n, w_n_1, I_n, I_n_1):
    """E_l - Image data likelihood (equation 9)
    
    x_r
        a rig site
    x_r_prime
        rig site in next frame based off candidate interpolation
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
    x_r_prime
        rig site in next frame based off candidate interpolation
    w_n_1: w_n-1
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    candidate: d^h_n,n-1
        vector estimating the motion of the hidden area
    d_prev: d_n-1,n-2
        vector matrix with the motion mapping from frame n-1 to frame n-2
    """
    temp = (1. / SIGMA_V ** 2.) * \
           w_n_1[x_r_prime] * \
           linalg.norm(candidate - d_prev[x_r_prime]) ** 2.

    return temp

def neighborhood((y, x), (height, width)):
    """Returns coordinates in the neighborhood of (y, x) omitting coordinates that go outside
    the bounds of (height, width)."""
    return [(yt, xt) for xt in [x + 1, x, x - 1]
                     for yt in [y + 1, y, y - 1]
                     if 0 <= xt < width and 0 <= yt < height
                     and (xt, yt) != (x, y)]

def lambda_(s, x_r):
    """Weights against points farther away from x_r. As defined in section 3 of the paper."""
    return LAMBDA / linalg.norm(np.array(s) - np.array(x_r))

def E_s(x_r, candidate, d_h):
    """E_s - Spatial motion smoothness (equation 9)"""
    temp = sum(lambda_(s, x_r) * linalg.norm(candidate - d_h[s]) ** 2.
               for s in neighborhood(x_r, d_h.shape[:2]))

    return temp

def E_0_o(x_r, occlusion):
    """E_0_o - Spatial occlusion smoothness (equation 9)"""
    temp = sum(lambda_(s, x_r) * abs(occlusion[s]) ** 2.
               for s in neighborhood(x_r, occlusion.shape[:2]))

    return temp

def E_1_o(x_r, occlusion):
    """E_1_o - Spatial occlusion smoothness (equation 9)"""
    temp = sum(lambda_(s, x_r) * abs(1. - occlusion[s]) ** 2.
               for s in neighborhood(x_r, occlusion.shape[:2]))

    return temp


#
# Main 
#

def reconstruct_frame(displacement, d_prev, vertices, w_n, w_n_1, I_n, I_n_1):
    """Reconstructs the given frame I_n, returning the new version along with updated weights.
    
    displacement - d_n,n-1, optical flow for the non rig areas of the current frame
    d_prev - d_n-1,n-2, optical flow mapping from the previous frame to the frame before that.
        Used for interpolation.
    vertices - vertices of the polygon outlining the rig region
    w_n - the weights for the current frame
    w_n_1 - w_n-1, the weights for the previous frame
    I_n - image intensity for the current frame
    I_n_1 - I_n-1, image intensity for the previous frame
    """
    shape = I_n.shape
    bounds = bounding_box(vertices, shape)
    
    # calculate spatial interpolation vector (section 4.2)
    siv = spatial_interpolation_vector(displacement, vertices)
    
    # initialize the candidates for the motion with the spatial interpolation
    candidates = np.empty(shape, dtype=object)
    for index, y in np.ndenumerate(candidates):
        candidates[index] = [siv]
    
    # find temporal interpolation candidates (section 4.3)
    temporal_interpolation_vectors(d_prev, candidates, bounds)
    
    # add adjacent neighbors as candidates if they've been assigned
    for x_r in index_iterator(bounds):
        if w_n[x_r] == 1:
            candidate = displacement[x_r]
            for s in neighborhood(x_r, shape):
                if w_n[s] < 1:
                    candidates[s].append(candidate)
    
    # candidate evaluation (section 4.4)
    # these are the initial candidates for occlusion and the hidden displacement
    occluded = np.logical_not(rig_matte(shape, vertices, dtype=bool))
    perturb = np.random.randn(*d_prev.shape) / 6.
    d_h = np.where(np.dstack((occluded, occluded)),
                   np.tile(siv, list(shape) + [1]) + perturb,
                   displacement)
    
    # the updated occlusion and displacement values will be placed in these
    new_occluded = occluded.copy()
    new_d_h = d_h.copy()
    is_rig = occluded.copy()
    for x_r in index_iterator(bounds):
        if is_rig[x_r]:
            minimum_energy = float('inf')
            best_candidate = None # the candidate associated with the minimum energy
            best_is_occluded = None
            
            # find the candidate that minimizes the local energy
            for candidate in candidates[x_r]:
                # the motion compensated site x_r' = x_r + d^h_n,n-1(x_r)
                x_r_prime = tuple((np.array(x_r) + candidate).round())
            
                el = E_l(x_r, x_r_prime, w_n, w_n_1, I_n, I_n_1)
                e0t = E_0_t(x_r, x_r_prime, w_n_1, candidate, d_prev)
                e1t = ALPHA
                es = E_s(x_r, candidate, d_h)
                e0o = E_0_o(x_r, occluded)
                e1o = E_1_o(x_r, occluded)
            
                e0 = el + e0t + es + e0o
                e1 = el + e1t + es + e1o
            
                # energy components determine whether the new displacement is occluded or not
                if e0 < minimum_energy:
                    minimum_energy = e0
                    best_candidate = candidate
                    best_is_occluded = False
                if e1 < minimum_energy:
                    minimum_energy = e1
                    best_candidate = candidate
                    best_is_occluded = True
        
            if best_candidate is not None:
                new_d_h[x_r] = best_candidate
                new_occluded[x_r] = best_is_occluded
    
    print 'occluded changed:', (occluded != new_occluded).sum()
    print 'd_h changed:', (d_h != new_d_h).sum()
    occluded = new_occluded
    d_h = new_d_h
    
    # construct the new intensities and weights based upon the computed motion vectors
    # using linear interpolation
    I_h = I_n.copy()
    for x_r in index_iterator(bounds):
        if is_rig[x_r]:
            x_r_prime = tuple((np.array(x_r) + d_h[x_r]).round())
            I_h[x_r] = (w_n[x_r] * I_n[x_r] + w_n_1[x_r_prime] * I_n_1[x_r_prime]) / (w_n[x_r] + w_n_1[x_r_prime])
    
    w_h = w_n.copy()
    for x_r in index_iterator(bounds):
        if is_rig[x_r]:
            x_r_prime = tuple((np.array(x_r) + d_h[x_r]).round())
            w_h[x_r] = (w_n[x_r] + w_n_1[x_r_prime]) / 2.
    
    return I_h, w_h
    

if __name__ == '__main__':
    # This is the main function driving everything.
    
    # load the images
    print 'loading images'
    load = lambda fname: ndimage.imread(fname, flatten=True) / 255.
    image_dir = 'Forest_Gump'
    files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
    images = [load(f) for f in files]
    
    shape = images[0].shape
    
    # load the rig vertices
    print 'loading vertices'
    vertices_list = parse_rig_vertices(file('rig_data.txt'))
    weights = [rig_matte(shape, v) for v in vertices_list]
    
    # load the displacements
    print 'loading displacements'
    displacement_dir = 'displacement'
    displacement_names = set(re.findall(r'\d{3}_\d{3}', f)[0]
                             for f in os.listdir(displacement_dir) if f.endswith('.csv'))
    displacements = [load_d(os.path.join(displacement_dir, dn))
                     for dn in sorted(displacement_names)]

    # loop over the images progressively reconstructing them
    print 'reconstructing frames'
    for i in xrange(len(images)-2):
        print '\nreconstructing frame', i
        
        # reconstruct the frame
        im, w_h = reconstruct_frame(
            displacements[i+1],
            displacements[i],
            vertices_list[i+2],
            weights[i+2],
            weights[i+1],
            images[i+2],
            images[i+1])
        
        # save out an image and update the stored image intensities and weights
        save_image('out/%03d.png' % (i + 2), im)
        images[i+2] = im
        weights[i+2] = w_h
