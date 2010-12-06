function nbh = neighborhood(x_r)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4
    
    (x,y) = x_r;
    nbh= [x-1, y-1; x, y - 1; x + 1, y - 1;
            x - 1, y; x + 1, y;
            x - 1, y + 1; x, y + 1; x + 1, y + 1];

end

