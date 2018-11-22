Thin wrapper around dlib's awesome face detection and alignment

The code is written to be minimalistic and without any unecessary dependencies besides dlib. Tested in Win 10 x64 and Ubuntu 16.04, both with MATLAB 2016b.

Data for the shape predictor can be downloaded at <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>. Without it, only the face detection in form of bounding boxes will work.

The function `extractroi` allows for stable region of interest (ROI) extraction via least squares shape alignment. This is useful e.g. for lipreading applications. See `test_image.m` for example usage.

If you want to build the mex file from source, you first need to compile the dlib library. On Linux, if [compiled as a static library](http://dlib.net/compile.html), make sure to add `-fPIC` option to `CXXFLAGS` before cmake configuration like so
```
CXXFLAGS=-fPIC cmake ..
```
otherwise the mex compilation will most likely fail. This is not needed on Windows with Visual Studio 2015 x64.
