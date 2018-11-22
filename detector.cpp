#include <fstream>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h> 
#include "mex.h"
#include "class_handle.hpp"

// taken from dlib's mex_wrapper.cpp: line 590 and slightly modified
void assign_mex_image_rgb(
    dlib::array2d<unsigned char>& img,
    const dlib::uint8* data,
    long nr,
    long nc
)
{
    const dlib::uint8 *pr = data;
    const dlib::uint8 *pg = data + nr * nc;
    const dlib::uint8 *pb = data + 2 * nr * nc;
    
    img.set_size(nr, nc);
    for ( long c = 0; c < img.nc(); ++c )
        for ( long r = 0; r < img.nr(); ++r ) {
            // When in RGB mode, dlib by default uses average of the channels
            // rather than the standard rgb2gray formula
            long gray = ( long(*pr++) + long(*pg++) + long(*pb++) ) / 3;
            img[r][c] = static_cast<dlib::uint8>( gray );
        }
}

// taken from dlib's mex_wrapper.cpp: line 590 and slightly modified
void assign_mex_image_gray(
    dlib::array2d<unsigned char>& img,
    const dlib::uint8* data,
    long nr,
    long nc
)
{
    img.set_size(nr, nc);
    for ( long c = 0; c < img.nc(); ++c )
        for ( long r = 0; r < img.nr(); ++r )
            img[r][c] = *data++;
}


class Detector
{
public:
    Detector();
    ~Detector();
    
    void load_predictor( const mxArray *str );
    void assign_image( const mxArray *image );
    void detect( const mxArray *image, mxArray **dets );
    void fit( const mxArray *image, const mxArray *rect, mxArray **shape );
    void mean_shape( mxArray **shape );
    
private:
    dlib::frontal_face_detector _fd;
    dlib::shape_predictor *_sp;
    dlib::matrix<float, 0, 1> _mean_shape;
    dlib::array2d<unsigned char> _img;
    std::vector<dlib::rectangle> _dets;
};

Detector::Detector()
    : _fd(dlib::get_frontal_face_detector()),
    _sp(NULL)
{
    
}

Detector::~Detector()
{
    if ( _sp ) {
        delete _sp;
        _sp = NULL;
    }
}

void Detector::load_predictor( const mxArray *str )
{
    char spfile[1024];
    if ( mxGetString( str, spfile, sizeof(spfile) ) )
        mexErrMsgTxt( "Error parsing shape predictor filename string" );
    
    std::ifstream fin( spfile, std::ios::binary );
    if ( !fin.is_open() ) {
        mexErrMsgTxt( "Error opening shape predictor file" );
    }
    
    if ( _sp )
        delete _sp;
    
    _sp = new dlib::shape_predictor();
    
    try {
        dlib::deserialize( *_sp, fin );
    }
    catch ( ... ) {
        delete _sp;
        _sp = NULL;
        _mean_shape.set_size(0);
        mexErrMsgTxt( "Error reading shape predictor file" );
    }
    
    // Read again the mean shape (it is a private member of dlib::shape_predictor,
    // so it cannot be accessed)
    fin.clear();
    fin.seekg(0);
    int version = 0;
    dlib::deserialize( version, fin );
    dlib::deserialize( _mean_shape, fin );
}

void Detector::assign_image( const mxArray *image )
{
    const mwSize *dims = mxGetDimensions( image );
    const dlib::uint8* imptr = reinterpret_cast<const dlib::uint8*>( mxGetData(image) );
    
    // If empty image, use the cached one (so we don't have to convert the same image multiple times)
    const mwSize nb_dims = mxGetNumberOfDimensions( image );
    if ( nb_dims == 2 && dims[0] == 0 && dims[1] == 0 )
        return;
    
    if ( nb_dims < 2 || nb_dims > 3 )
        mexErrMsgTxt( "uint8 grayscale or RGB image required" );
    
    if ( !mxIsUint8( image ) )
        mexErrMsgTxt( "Data must be real of type uint8" );
        
    // Image must be converted from MATLAB's column major to dlib's row major mode
    if ( nb_dims == 2 )
        assign_mex_image_gray( _img, imptr, dims[0], dims[1] );
    else
        assign_mex_image_rgb( _img, imptr, dims[0], dims[1] );
}

void Detector::detect( const mxArray *image, mxArray **dets )
{   
    if ( !image )
        mexErrMsgTxt( "No input image provided" );
    
    assign_image( image );
    
    std::vector<dlib::rectangle> faces = _fd(_img);
    int nf = faces.size();
    
    if ( nf == 0 ) {
        mexPrintf( "No face detected.\n" );
        *dets = mxCreateDoubleMatrix( 0, 0, mxREAL );
        return;
    }
        
    *dets = mxCreateDoubleMatrix( nf, 4, mxREAL );
    double *rs = reinterpret_cast<double*>( mxGetData( *dets ) );
    for ( int i = 0; i < nf; ++i ) {
        rs[i] = faces[i].left();
        rs[i + nf] = faces[i].top();
        rs[i + 2 * nf] = faces[i].right();
        rs[i + 3 * nf] = faces[i].bottom();
    }
}

void Detector::fit( const mxArray *image, const mxArray *rect, mxArray **shape )
{
    assign_image( image );
    
    const double *pr = reinterpret_cast<double*>( mxGetData(rect) );
    dlib::rectangle det( pr[0], pr[1], pr[2], pr[3] );
    
    if ( !_sp ) {
        mexErrMsgTxt( "Shape predictor not loaded" );
    }
    
    dlib::full_object_detection fod = (*_sp)( _img, det );
    int np = fod.num_parts();
    
    if ( np == 0 ) {
        mexPrintf( "align error\n" );
        *shape = mxCreateDoubleMatrix( 0, 0, mxREAL );
        return;
    }
    
    *shape = mxCreateDoubleMatrix( np, 2, mxREAL );
    double *ps = reinterpret_cast<double*>( mxGetData( *shape ) );
    for ( int i = 0; i < np; ++i ) {
        ps[i] = fod.part(i).x();
        ps[i + np] = fod.part(i).y();
    }
}

void Detector::mean_shape( mxArray **shape )
{
    int nr = _mean_shape.nr() / 2;
    *shape = mxCreateDoubleMatrix( nr, 2, mxREAL );
    double *ps = reinterpret_cast<double*>( mxGetData( *shape ) );
    for ( int i = 0; i < nr; ++i ) {
        ps[i] = _mean_shape( 2 * i );
        ps[i + nr] = _mean_shape( 2 * i + 1 );
    }
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    // Get the command string
    char cmd[64];
	if ( nrhs < 1 || mxGetString( prhs[0], cmd, sizeof(cmd) ) )
		mexErrMsgTxt( "First input should be a command string less than 64 characters long." );
        
    // New
    if ( !strcmp( "new", cmd ) ) {
        // Check parameters
        if ( nlhs != 1 )
            mexErrMsgTxt( "New: One output expected." );
        
        Detector *det = new Detector();
        if ( nrhs > 1 ) {
            det->load_predictor( prhs[1] );
        }
        
        // Return a handle to a new C++ instance
        plhs[0] = convertPtr2Mat<Detector>( det );
        return;
    }
    
    // Check there is a second input, which should be the class instance handle
    if ( nrhs < 2 )
		mexErrMsgTxt( "Second input should be a class instance handle." );
    
    // Delete
    if ( !strcmp( "delete", cmd ) ) {
        // Destroy the C++ object
        destroyObject<Detector>( prhs[1] );
        // Warn if other commands were ignored
        if ( nlhs != 0 || nrhs != 2 )
            mexWarnMsgTxt( "Delete: Unexpected arguments ignored." );
        return;
    }
    
    Detector *det = convertMat2Ptr<Detector>( prhs[1] );
    
    // Load shape predictor data
    if ( !strcmp( "load_predictor", cmd ) ) {
        det->load_predictor( prhs[2] );
        return;
    }
    
    // Detect faces
    if ( !strcmp( "detect", cmd ) ) {
        det->detect( prhs[2], &plhs[0] );
        return;
    }
    
    // Fit precise shape
    if ( !strcmp( "fit", cmd ) ) {
        det->fit( prhs[2], prhs[3], &plhs[0] );
        return;
    }
    
        // Return the reference shape
    if ( !strcmp( "mean_shape", cmd ) ) {
        det->mean_shape( &plhs[0] );
        return;
    }
}

