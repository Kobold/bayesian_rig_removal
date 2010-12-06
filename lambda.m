function l_s = lambda(s,x_r)
    % The weight associated with each clique
    % Discourages 'smoothness' over too large a range
    SIGMA_E = 0.01; % allowance of acceleration
    SIGMA_V = 0.01; % allowance of acceleration
    LAMBDA = 2.0;
    ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4
    
    l_s= LAMBDA / norm(s - x_r);
end

