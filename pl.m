function plr = pl(x_r, w_n, w_n_1, d_h, I_n, I_n_1)
    %Image data likelihood

    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4

    % x_r: x_r
    % a rig site
    % w_n: w_n
    % continuous matrix. 1 indicates data available, 0 indicates data missing.
    % in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    % w_n_1: w_n-1
    % continuous matrix. 1 indicates data available, 0 indicates data missing.
    % in the rig area w(x_r) = 0, i.e. this is the "not-rig" matrix
    % d_h: d^h_n,n-1; vector format (ie. [x y])
    % vector matrix estimating the motion of the hidden area
    % I_n: I_n
    % frame at n
    % I_n_1: I_n-1
    % frame at n-1
    
    x_r_prime = x_r + d_h(x_r);

    temp = (1 / (2 * (SIGMA_E ^ 2))) * ...
           w_n(x_r) * ...
           w_n_1(x_r_prime) * ...
           ((I_n(x_r) - I_n_1(x_r_prime)) ^ 2);
           
    plr = exp(-temp);
    

end

