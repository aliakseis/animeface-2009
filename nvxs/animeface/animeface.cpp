#include <nv_core.h>
#include <nv_ip.h>
#include <nv_ml.h>
#include <nv_face.h>

//#include <opencv/cv.h>
#define HAVE_OPENCV_IMGCODECS
#include <opencv2/highgui.hpp>

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>

//static VALUE cMagickPixel;
//static VALUE cMagick;

#define NV_MAX_FACE 4096
#define NV_BIT_SCALE(bits)  (1.0f / (powf(2.0f, bits) - 1.0f))
#define NV_RMAGICK_DATA_TO_8BIT_FLOAT(x, depth) ((float)(NUM2DBL(x)) * NV_BIT_SCALE(depth) * 255.0f)
#define NV_TO_RMAGICK_DATA(x, depth) (x * 1/255.0f * pow(2.0, depth))

#define NV_ANIMEFACE_WINDOW_SIZE  42.592f
#define NV_ANIMEFACE_STEP         4.0f
#define NV_ANIMEFACE_SCALE_FACTOR 1.095f
#define NV_ANIMEFACE_THRESHOLD    0.5f

auto rotate_bound(const cv::Mat& image, double angle)
{
    // grab the dimensions of the image and then determine the
    // center
    // (h, w) = image.shape[:2]
    int h = image.rows;
    int w = image.cols;
    // (cX, cY) = (w // 2, h // 2)
    int cX = w / 2;
    int cY = h / 2;
    // grab the rotation matrix(applying the negative of the
    // angle to rotate clockwise), then grab the sine and cosine
    // (i.e., the rotation components of the matrix)
    auto M = cv::getRotationMatrix2D(cv::Point2f( cX, cY ), -angle, 1.0);
    auto cos = std::abs(M.at<double>(0, 0));
    auto sin = std::abs(M.at<double>(0, 1));
    // compute the new bounding dimensions of the image
    auto nW = int((h * sin) + (w * cos));
    auto nH = int((h * cos) + (w * sin));
    // adjust the rotation matrix to take into account translation
    M.at<double>(0, 2) += (nW / 2) - cX;
    M.at<double>(1, 2) += (nH / 2) - cY;
    // perform the actual rotation and return the image
    cv::Mat dst;
    cv::warpAffine(image, dst, M, cv::Size(nW, nH));
    return dst;
}


static void
nv_conv_imager2nv(nv_matrix_t *bgr, nv_matrix_t *gray,
    cv::Mat& im)
{
    const auto xsize = im.cols;
    const auto ysize = im.rows;

    for (int y = 0; y < ysize; ++y) {
        for (int x = 0; x < xsize; ++x) {
            const auto& c = im.at<cv::Vec3b>(y, x); //rb_funcall(im, rb_id_pixel_color, 2, INT2FIX(x), INT2FIX(y));
            const auto b = c[0]; // NV_RMAGICK_DATA_TO_8BIT_FLOAT(rb_funcall(c, rb_id_green, 0), depth);
            const auto g = c[1]; //NV_RMAGICK_DATA_TO_8BIT_FLOAT(rb_funcall(c, rb_id_blue, 0), depth);
            const auto r = c[2]; //NV_RMAGICK_DATA_TO_8BIT_FLOAT(rb_funcall(c, rb_id_red, 0), depth);

            NV_MAT3D_V(bgr, y, x, NV_CH_B) = b;
            NV_MAT3D_V(bgr, y, x, NV_CH_G) = g;
            NV_MAT3D_V(bgr, y, x, NV_CH_R) = r;
        }
    }
    IplImage *cvBgr = nv_to_image(bgr);
    IplImage *cvGray = nv_to_image(gray);
    cvCvtColor(cvBgr, cvGray, CV_BGR2GRAY);
    cvReleaseImageHeader(&cvBgr);
    cvReleaseImageHeader(&cvGray);
}

struct DetectResult {
    nv_face_feature_t feature;
    nv_face_position_t pos;
};


//VALUE detect(VALUE im,
auto detect(cv::Mat im,
			 float min_window_size,
			 float step, float scale_factor,
			 float threshold)
{
	static const nv_mlp_t *detector_mlp = &nv_face_mlp_face_00;
	static const nv_mlp_t *face_mlp[] = {
		&nv_face_mlp_face_01,
		&nv_face_mlp_face_02,
		nullptr
	};
	static const nv_mlp_t *dir_mlp = &nv_face_mlp_dir;
	static const nv_mlp_t *parts_mlp = &nv_face_mlp_parts;

    std::vector<DetectResult> results;
	
	CvRect image_size;
	nv_face_position_t face_pos[NV_MAX_FACE];
	
    const int xsize = im.cols;
    const int ysize = im.rows;

	nv_matrix_t *bgr = nv_matrix3d_alloc(3, ysize, xsize);
	nv_matrix_t *gray = nv_matrix3d_alloc(1, ysize, xsize);
	nv_matrix_t *gray_integral = nv_matrix3d_alloc(1, ysize + 1, xsize + 1);
	nv_matrix_t *edge_integral = nv_matrix3d_alloc(1, ysize + 1, xsize + 1);

	// initialize
	
	nv_matrix_zero(bgr);
	nv_matrix_zero(gray);
	nv_matrix_zero(gray_integral);
	nv_matrix_zero(edge_integral);
	
	image_size.x = image_size.y = 0;
	image_size.width = gray->cols;
	image_size.height = gray->rows;
	
	// convert format
	nv_conv_imager2nv(bgr, gray, im);
	IplImage *cvGray = nv_to_image(gray);


	// edge
	IplImage *cvEdge = cvCreateImage(cvSize(xsize, ysize), IPL_DEPTH_32F, 1);
	cvSmooth(cvGray, cvEdge, CV_GAUSSIAN, 5, 5, 1, 0);
	cvLaplace(cvEdge, cvEdge, 1);
    for (int x = 0; x < xsize; ++x) {
        for (int y = 0; y < ysize; ++y) {
            if (CV_IMAGE_ELEM(cvEdge, float, y, x) < 0) {
                CV_IMAGE_ELEM(cvEdge, float, y, x) = 0;
            }
            else {
                CV_IMAGE_ELEM(cvEdge, float, y, x) *= 4;
            }
        }
    }
	nv_matrix_t *edge = nv_from_image(cvEdge);
	// integral
	nv_integral(gray_integral, gray, 0);
	nv_integral(edge_integral, edge, 0);
	
	// detect face
	int nface = nv_face_detect(face_pos, NV_MAX_FACE,
						   gray_integral, edge_integral, &image_size,
						   dir_mlp,
						   detector_mlp, face_mlp, 2,
						   parts_mlp,
						   step, scale_factor, min_window_size, threshold
		);
	// analyze face 
	for (int i = 0; i < nface; ++i) {
		nv_face_feature_t face_feature = {0};
		nv_face_analyze(&face_feature, &face_pos[i], bgr);
		
		/*
		 * likelihood => ,
		 * face => {x=>,y=>,width=>,height=>, skin_color =>, hair_color=>, },
		 * eyes => {left} => { x=>,y=>,width=>,height=> , color => (,,,,)},
		 * eyes => {right} => { x=>,y=>,width=>,height=> , color => (,,,,)}, 
		 * nose => {x=>, y=>, width=>1, height=>1},
		 * mouse => { x=>, y=>, width=>, height=>},
		 * chin => { x=>, y =>, width=>1, height=>1}
		 */
		
        results.push_back({ face_feature, face_pos[i] });		
	}
	nv_matrix_free(&bgr);
	nv_matrix_free(&gray);
	nv_matrix_free(&edge);
	nv_matrix_free(&gray_integral);
	nv_matrix_free(&edge_integral);

	cvReleaseImageHeader(&cvGray);
	cvReleaseImage(&cvEdge);
	
	return results;
}


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: animeface input_file\n";
        return EXIT_FAILURE;
    }

    cv::Mat img = rotate_bound(cv::imread(argv[1]), 0);

    auto results = detect(img, NV_ANIMEFACE_WINDOW_SIZE, NV_ANIMEFACE_STEP, NV_ANIMEFACE_SCALE_FACTOR, NV_ANIMEFACE_THRESHOLD);

    for (const auto& result : results)
    {
        cv::Rect rct(result.pos.face);
        cv::Scalar clr{ 255, 0, 255 };
        cv::rectangle(img, rct, clr);
    }

    cv::namedWindow("result", cv::WINDOW_NORMAL);
    cv::imshow("result", img);

    cv::waitKey();

    return 0;
}
