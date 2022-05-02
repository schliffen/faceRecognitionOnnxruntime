//
// Created by ai on 5/24/21.
//

#ifndef EXAMPLE_1_FACEPROCESS_H
#define EXAMPLE_1_FACEPROCESS_H


#include <vector>
#include <iostream>
#include <string.h>
#include <map>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include <onnxruntime_cxx_api.h>

#define SMALL_NUM 1e-12

typedef float fp_t;
typedef double dt_p;

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int batchOrder;
    float landmark5[10];
    // Note: feature size 512
    float features[512];

} FaceInfo;


struct ref_landmark {

    dt_p org_landmark_1[5][2] = {
            { 30.2946, 51.6963 },
            { 65.5318, 51.5014 },
            { 48.0252, 71.7366 },
            { 33.5493, 92.3655 },
            { 62.7299, 92.2041 }
    };

    dt_p ref_mean[2] = { 48.02616, 71.90078 };
    dt_p std = 13.637027193284474;

    dt_p standardized[5][2] = {
            { -1.7731562e+01, -2.0204479e+01 },
            { 1.7505638e+01, -2.0399380e+01 },
            { -9.6130371e-04, -1.6417694e-01 },
            { -1.4476860e+01,  2.0464722e+01 },
            { 1.4703739e+01,  2.0303322e+01 }
    };

}reference_landmark;


class faceprocess{
    public:
        static const int det_inWidth      = 320;
        static const int det_inHeight     = 320;
        std::vector<float> det_inputTensorValues;

        // ------ feature part
        const int feat_inWidth      = 112;
        const int feat_inHeight     = 112;
        const int feature_size = 512;
        int numOutputNodes, det_batch_size, feat_batch_size=1;
        const float feat_inScaleFactor[2] = {1,0};//{0.00784314,1};
        fp_t min_sizes[3][2] = { { 16, 32 }, { 64, 128 }, { 256, 512 } };
        fp_t steps[3] = { 8, 16, 32 };
        dt_p variance[2] = { 0.1, 0.2 };



        size_t det_input_tensor_size = det_inWidth * det_inHeight * 3;
        size_t feat_input_tensor_size = feat_inWidth * feat_inHeight * 3;
        //const char* model_path;
        std::vector<const char *> det_output_node_names;
        std::vector<const char*> det_input_node_names; // in case of multiple names
        std::vector<int64_t> det_input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
        std::vector<Ort::Value> det_inputTensors;

        std::vector<const char *> feat_output_node_names;
        std::vector<const char*> feat_input_node_names; // in case of multiple names
        std::vector<int64_t> feat_input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
        std::vector<Ort::Value> feat_inputTensors;

        std::vector<long, std::allocator<long> > det_output_node_dims;
        std::vector<long, std::allocator<long> > feat_output_node_dims;

        std::vector<float> feat_inputTensorValues;

        std::unique_ptr<Ort::Env> fdet_env = NULL;
        std::unique_ptr<Ort::Session> fdet_session = NULL;
        std::unique_ptr<Ort::MemoryInfo> det_memory_info = NULL;
        //
        std::unique_ptr<Ort::Env> adet_env = NULL;
        std::unique_ptr<Ort::Session> feat_session = NULL;
        std::unique_ptr<Ort::MemoryInfo> feat_memory_info = NULL;

        float score_threshold = .5;
        float iou_threshold = .3;
        const float det_mean_vals[3] = {104.0f, 117.0f, 123.0f};
        std::vector<std::vector<float>> anchors;
        std::vector<FaceInfo> detectedFaces;


    public:
        faceprocess(const char *, const char * );
        std::vector<cv::Mat> set_input( std::vector<cv::Mat>, char * task );
        void do_forward( );
        // tiny detector
        void generateBBox(std::vector<FaceInfo> &bbox_collection, float* scores, float* boxes );
        void tiny_nms(std::vector<FaceInfo> input);
        void get_features( );
        // retina detector

        void calculateAnchors();
        void get_feature_map(int16_t INPUT_H, int16_t INPUT_W, int feature_maps[3][2]);
        std::vector<FaceInfo> decode_bboxes(float *bboxes, float *confidence, float *landmarks );
        std::vector<cv::Mat> imgsToFlattenedVector(std::vector<cv::Mat>& imgs, float * out);
        int batchOptimizedWrappFlatten( std::vector<float>&srcs, std::vector<float>& detectedFaces);

};


bool transformation_matrix( fp_t flandmarks[10],fp_t M[2][3]);
bool invert_indx( int indi, int indj, dt_p rindex[2], fp_t M[2][3]);




#endif //EXAMPLE_1_FACEPROCESS_H
