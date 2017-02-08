function [ r, varargout ] = extractroi( img, shape, params, varargin )
% EXTRACTROI Extracts region of interest (ROI) based on aligning input and
% reference shapes
%   Input:
%       img ... input image
%       shape ... an n_landmarks x 2 array of (x, y) pairs
%       params ... structure containing parameters of the extractor; see
%                  INITROI
%       [T] ... optional input transformation; if not provided, it's
%               calculated automatically using `fitgeotrans`
%   Returns:
%       r ... output ROI matrix of the same type as input image; the size
%             is defined by the sz attribute of the params struct
%       [T] ... 2d geometric transformation computed by aligning input and
%               reference shapes

% interp2 only works with double precision
dtype = class(img);
if ~isa(img, 'double')
    img = double(img);
end

sz = size(img);   

% Either get transformation from user or estimate it from shapes
if nargin > 3
    tform = varargin{1};
else
    tform = fitgeotrans(squeeze(shape), squeeze(params.ref_shape), params.ttype);
end

% Transform whole reference meshgrid into image coordinates: these will be
% the interpolation points
[u, v] = transformPointsInverse(tform, params.x, params.y);

% Interpolate
if length(sz) == 3
    r = zeros(params.sz(1), params.sz(2), sz(3));
    for c = 1:sz(3)
        r(:, :, c) = interp2(img(:, :, c), u, v);
    end
elseif length(sz) == 2
    r = interp2(img, u, v);
else
    error('extractroi: image dimensions not understood');
end

% Convert back to original dtype if necessary
if ~isa(r, dtype)
    r = cast(r, dtype);
end

varargout{1} = tform;

end

