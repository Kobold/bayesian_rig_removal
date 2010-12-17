"""
how do we update the occlusion data?
where does the weight field come from,

how is the estimate worked into the main estimation--there's overlap in terms

do we work from back to front

where the fuck does lambda_o in equation 6 come from?
"""

SIGMA_E = 0.01 # allowance of acceleration
SIGMA_V = 0.01 # allowance of acceleration
LAMBDA = 2.0
ALPHA = 0.0331 # ~ 3 * SIGMA_V according to section 3.4


#
# Helpers
#

def neighborhood(x_r):
    """The function S_n. Returns the eight points surrounding x_r."""
    x, y = x_r
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y),                 (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

def lambda_(s, x_r):
    """Discourages 'smoothness' over too large a range."""
    return LAMBDA / magnitude(s - x_r)

#
# Probability model components
#

def p_l(x_r, w_n, w_n_1, d_h, I_n, I_n_1):
    """
    x_r: x_r
        a rig site
    w_n: w_n
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    w_n_1: w_n-1
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    d_h: d^h_n,n-1
        vector matrix estimating the motion of the hidden area
    I_n: I_n
        frame at n
    I_n_1: I_n-1
        frame at n-1
    """
    # the motion compensated site x_r + d^h_n,n-1(x_r)
    x_r_prime = x_r + d_h(x_r)

    temp = (1. / (2. * (SIGMA_E ** 2.))) *
           w_n(x_r) *
           w_n_1(x_r_prime) *
           ((I_n(x_r) - I_n_1(x_r_prime)) ** 2)
           
    return exp(-temp)


def p_t(x_r, occlusion, data_available, d_estimate, d_prev):
    """
    x_r:
        a rig site
    occlusion: o_n,n-1
        binary matrix. 1 indicates data that that point in the frame n does not
        exist in the frame n-1. 0 indicates no discontinuity
    data_available: w_n-1
        continuous matrix. 1 indicates data available, 0 indicates data missing.
        in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    d_estimate: d^h_n,n-1
        vector matrix estimating the motion of the hidden area
    d_prev: d_n-1,n-2
        vector matrix with the motion mapping from frame n-1 to frame n-2
    """
    # the motion compensated site x_r + d^h_n,n-1(x_r)
    x_r_prime = x_r + d_estimate(x_r)
    
    temp = (1. / (SIGMA_V ** 2.)) *
           (1. - occlusion(x_r)) *
           data_available(x_r_prime) *
           (magnitude(d_estimate(x_r) - d_prev(x_r_prime)) ** 2.)
    
    return exp(-temp)


def p_s(x_r, d_estimate, d):
    """
    x_r: rig site
    d_estimate: vector matrix estimating motion 
        of the hidden area, as in other functions
    d: matrix of motion vectors (maybe - section 4.1)

    """
    temp = sum(lambda_(s, x_r) * (magnitude(d_estimate(x_r) - d(s)) ** 2.)
               for s in neighborhood(x_r))
    
    return exp(-temp)
    

def p_so(x_r, occlusion):
    """
    x_r:
        a rig site
    occlusion: o_n,n-1
        binary matrix. 1 indicates data that that point in the frame n does not
        exist in the frame n-1. 0 indicates no discontinuity
    """
    occ = sum(lambda_(s, x_r) * magnitude(occlusion(x_r) - occlusion(s))
              for s in neighborhood(x_r))
    
    # TODO: is this sum term *actually* the sum of all the occlusion values?
    penalty = ALPHA * sum(occlusion)
    
    return exp(-acc) * exp(-penalty)
