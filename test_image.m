close all;

rgb = imread('2008_001322.jpg');
% rgb = imresize(rgb, 2);
gray = rgb2gray(rgb);
h = detector('new', 'shape_predictor_68_face_landmarks.dat');
f = detector('detect', h, rgb);
m = detector('mean_shape', h);

params = initroi(m, [64, 64], [0.25, 0.6], [0.75, 0.9], 'similarity');

r = zeros(size(f, 1), 64, 64, 3, 'uint8');
s = zeros(size(f, 1), 68, 2);
for i = 1:size(f, 1)
    s(i, :, :) = detector('fit', h, rgb, f(i, :));
    r(i, :, :, :) = extractroi(rgb, s(i, :, :), params);
end

% show face detections and alignments
figure(1);
imshow(rgb);
hold on;
for i = 1:size(f, 1)
    x = [f(i, 1), f(i, 3), f(i, 3), f(i, 1), f(i, 1)];
    y = [f(i, 2), f(i, 2), f(i, 4), f(i, 4), f(i, 2)];
    plot(x, y, '-r');
    plot(s(i, :, 1), s(i, :, 2), 'g+');
end

% show the mean shape
figure(2);
plot(m(:, 1), m(:, 2), 'go');
for j = 1:size(m, 1)
    text(m(j, 1), m(j, 2), sprintf('%i', j));
end
set(gca, 'Ydir', 'reverse');

% show the extract regions of interest
figure(3);
for i = 1:size(r, 1)
    subplot(1, size(r, 1), i);
    imshow(im2uint8(squeeze(r(i, :, :, :))));
end

detector('delete', h);
