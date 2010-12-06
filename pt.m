function ptr = pt(x_r, occlusion, w_n_1, d_h, d_prev)
    %Image data likelihood

    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4

    %x_r:
    %a rig site
    %occlusion: o_n,n-1
    %binary matrix. 1 indicates data that that point in the frame n does not
    %exist in the frame n-1. 0 indicates no discontinuity
    %data_available: w_n-1
    %continuous matrix. 1 indicates data available, 0 indicates data missing.
    %in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    %d_estimate: d^h_n,n-1
    %vector matrix estimating the motion of the hidden area
    %d_prev: d_n-1,n-2
    %vector matrix with the motion mapping from frame n-1 to frame n-2
    %x_r_prime = x_r + d_h(x_r);

    %the motion compensated site x_r + d^h_n,n-1(x_r)
    x_r_prime = x_r + d_estimate(x_r);
    
    temp = (1 / (SIGMA_V ^ 2)) * ...
           (1 - occlusion(x_r)) * ...
           w_n_1(x_r_prime) * ...
           ((d_h(x_r) - d_prev(x_r_prime)) ^ 2);
    
    ptr = exp(-temp);

end

