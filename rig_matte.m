function out = rig_matte(w, h, xv, yv)
% w, h - the dimensions of the output matrix
% xv, yv - vectors that specify the vertices of a polygonal rig region

xv = xv'; yv = yv';
xv = [xv ; xv(1)]; yv = [yv ; yv(1)]; % close the polygon

x_values = [1:w];
x = repmat(x_values, h, 1);
y_values = [1:h]';
y = repmat(y_values, 1, w);

in = inpolygon(x,y,xv,yv);
out = double(ones(h, w) - in);

return
