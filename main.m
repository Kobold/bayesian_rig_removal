function p = main(Im_1, Im_2, Im_3)
%Project Summary of this function goes here
%   Detailed explanation goes here

SIGMA_E = 0.01; % allowance of acceleration
SIGMA_V = 0.01; % allowance of acceleration
LAMBDA = 2.0;
ALPHA = 0.0331; % ~ 3 * SIGMA_V according to section 3.4

I_n = rgb2gray(Im_3);
I_n_1 = rgb2gray(Im_2);

%occlusion
[rows, cols] = size(Im_1);
occlusion = ones(rows, cols);

%d_prev
d_prev_x = textread('d_prev_x.csv', '', 'delimiter', ',', 'emptyvalue', NaN);
d_prev_y = textread('d_prev_y.csv', '', 'delimiter', ',', 'emptyvalue', NaN);
d_prev = [d_prev_x, d_prev_y];

% w_n is the weight field for frame 3
w_n = rig_matte(cols, rows, [679 719 742 680], [270 264 339 340]);

% w_n_1 is the weight field for frame 2
w_n_1 = rig_matte(cols, rows, [679 726 740 679], [273 263 334 337]);

%Code
for i = 1:rows
    for j = 1:cols
        x_r = [i, j];
        
        p = ((pl(x_r, w_n, w_n_1, d_prev, I_n, I_n_1).*pt(x_r, occlusion, w_n_1, d_prev, d_prev)) * ps(x_r, d_prev, d_prev) * pso(x_r, occlusion));
    end
end    

end