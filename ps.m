function psr = ps( x_r, d_h, d )
    % Spatial motion smoothness
    
    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4
    
    % x_r: rig site
    % d_estimate: vector matrix estimating motion
    % of the hidden area, as in other functions
    % d: matrix of motion vectors (maybe - section 4.1)
    temp=0;
    s=neighborhood(x_r);
    for n=1:size(n,1)
        temp = temp+ (lambda(s(n,:), x_r) * ...
            (norm(d_h(x_r) - d(s(n,:)))) ^ 2);
    end

    psr= exp(-temp);

end

