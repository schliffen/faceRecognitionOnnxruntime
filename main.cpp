#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/faceProcess.h"
#include "include/preprocessImg.h"

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main(int argc, char * argv[]) try
{
    const char* det_model_path = "/home/ai/EkinStash/ai-inference/FaceRecognition_Project/FaceModels/onnxmodel/retinaface_mb025_dynamic.onnx"; // retina facenet
    const char* align_model_path = "/home/ai/EkinStash/ai-inference/FaceRecognition_Project/FaceModels/onnxmodel/glintr100_dynamic.onnx"; // light weight detector


    faceprocess fdet( det_model_path,  align_model_path);

    std::string img_path = "/home/ai/EkinStash/ai-inference/common/images_d/*.ppm";
    std::vector<cv::String> fn;
    cv::glob (img_path, fn, false );
    size_t count = fn.size();

    std::vector<float> templates;
    std::vector<cv::Mat> tempFaces, inputImgs, inImgs, faces;

    for (int u=0; u<fn.size(); u++) {
        cv::Mat color_mat = cv::imread(fn.at(u));
        inputImgs.push_back( color_mat );
    }

    inImgs = fdet.set_input(inputImgs, "det");
    // detecting the face
    fdet.do_forward();
    fdet.set_input(inImgs, "feat");
    fdet.get_features();

    // control part
    std::vector<cv::Mat> facelist;
    for (int u=0; u<fdet.detectedFaces.size(); u++) {

        cv::Rect rect((int) fdet.detectedFaces[u].x1, (int) fdet.detectedFaces[u].y1,
                      (int) (fdet.detectedFaces[u].x2 - fdet.detectedFaces[u].x1),
                      (int) (fdet.detectedFaces[u].y2 - fdet.detectedFaces[u].y1));


        cv::rectangle(inImgs.at(fdet.detectedFaces[u].batchOrder), cv::Point((int) fdet.detectedFaces[u].x1, (int) fdet.detectedFaces[u].y1),
                      cv::Point((int) fdet.detectedFaces[u].x2, (int) fdet.detectedFaces[u].y2), cv::Scalar(10*u, 200, 1),
                      2);

        facelist.push_back(  inImgs.at(fdet.detectedFaces[u].batchOrder)(rect));


        for (int i = 0; i < 5; i++) {
            int lx = fdet.detectedFaces[u].landmark5[2 *
                                                     i]; //  * (fdet.detectedFaces[0].x2 - fdet.detectedFaces[0].x1) + fdet.detectedFaces[0].x1 ;
            int ly = fdet.detectedFaces[u].landmark5[2 * i +
                                                     1];
            cv::circle(inImgs.at(fdet.detectedFaces[u].batchOrder), cv::Point(lx, ly), 3, cv::Scalar(u*10, 20, 200), 3);
        }

        cv::imshow("inimg", inImgs.at(fdet.detectedFaces[u].batchOrder));
        cv::waitKey(0);

    };


    // comparing features
    cv::Mat comFaces,f1,f2;
    for (int i=0; i< fdet.detectedFaces.size(); i++){
        for (int j=0; j< fdet.detectedFaces.size(); j++){
            // comparing collected faces
            float score = 0;
            for (int k=0; k<512; k++)
                score += fdet.detectedFaces[i].features[k] * fdet.detectedFaces[j].features[k];
            cv::resize(facelist[i], f1, cv::Size(112,112));
            cv::resize(facelist[j], f2, cv::Size (112,112));

            cv::hconcat( f1, f2, comFaces);
            cv::putText(comFaces, "similarity: " + std::to_string(score) , cv::Point(10, 10), 1, 1, cv::Scalar(10, 200,20));
            cv::imshow( "compare", comFaces);
            cv::waitKey(0);

            std::cout << "score: " << score << std::endl;

        }

    }


    return EXIT_SUCCESS;
}

catch (const std::exception & e)
{
    std::cerr << "RealSense error calling " << e.what() << "(" << e.what() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
