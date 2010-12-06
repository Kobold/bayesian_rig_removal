function psor = pso(x_r, occlusion)
    %Spatial occlusion smoothness
    
    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4
    
    % x_r:
    % a rig site
    % occlusion: o_n,n-1
    % binary matrix. 1 indicates data that that point in the frame n does not
    % exist in the frame n-1. 0 indicates no discontinuity

    temp=0;
    s=neighborhood(x_r);
    for n=1:size(n,1)
        temp = temp+ (lambda(s(n,:), x_r) ...
            * norm(occlusion(x_r) - occlusion(s(n,:)));
    end

    % TODO: is this sum term *actually* the sum of all the occlusion values?
    penalty = ALPHA * sum(occlusion);
    
    psor= exp(-acc) * exp(-penalty);
end

