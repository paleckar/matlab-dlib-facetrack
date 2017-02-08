function [ params ] = initroi( ref_shape, sz, tl, br, ttype )
% INITROI Initializes parameters for region of interest (ROI) extraction
%   Input:
%       ref_shape ... an n_landmarks x 2 array of (x, y) pairs describing
%                     the reference object shape
%       sz ... target size of the ROI as [height, width] pair
%       tl ... top-left [x, y] point of the ROI rectangle in normalized
%              reference coordinates
%       br ... bottom-right [x, y] point of the ROI rectangle in normalized
%              reference coordinates
%       ttype ... transformation type; see MATLAB's fitgeotrans for
%                 list of possible values
%   Returns:
%       params ... struct passed to EXTRACTROI function

nw = br(1) - tl(1);
nh = br(2) - tl(2);
sx = nw / sz(2);
sy = nh / sz(1);
[x, y] = meshgrid(tl(1)+sx/2:sx:br(1), tl(2)+sy/2:sy:br(2));

params = struct;
params.ref_shape = ref_shape;
params.sz = sz;
params.tl = tl;
params.br = br;
params.ttype = ttype;
params.x = x;
params.y = y;

end

