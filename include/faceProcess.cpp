//
// Created by ai on 5/24/21.
//

#include "faceProcess.h"
// todo: this is only for test
#include <fstream>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

cv::Mat preprocess_img(cv::Mat img, int INPUT_W, int INPUT_H) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}


std::vector<cv::Mat> faceprocess::imgsToFlattenedVector(std::vector<cv::Mat>& Imgs, float * out) {

    // Create an Ort tensor containing random numbers for batch processing
    cv::Mat img;
    int imgSize = 0;
    det_batch_size=0;
    int singleJump = det_inHeight * det_inWidth;

    std::vector<cv::Mat> debugImg;

    for (int l = 0; l < Imgs.size(); l++) {
        img = preprocess_img(Imgs.at(l), det_inWidth, det_inHeight);
        debugImg.push_back(img);

        cv::Vec3b v;
        float blue, red, green;
        int redind, greenind, blueind;
        //
        for (int r = 0; r < det_inHeight; r++) {
            for (int q = 0; q < det_inWidth; q++) {
                v = img.at<cv::Vec3b>(r, q);
                blue = v[2]; // 2
                red = v[0]; // 0
                green = v[1]; // 1
                redind = r * det_inWidth + q;
                greenind = singleJump + r * det_inWidth + q;
                blueind = 2 * singleJump + r * det_inWidth + q;
                out[imgSize + redind] = (red - det_mean_vals[0]);//
                out[imgSize + greenind] = (green - det_mean_vals[1]);//
                out[imgSize + blueind] = (blue - det_mean_vals[2]); //

            }// for (int q = 0; q < cols; q++)
        }; // for (int r = 0; r < rows; r++)
        imgSize += 3*singleJump;
        det_batch_size++;

    } // (int l=0; l<imgs.size();l++)
    det_input_tensor_size = imgSize;

    return debugImg;

}




faceprocess::faceprocess(const char * det_path, const char * align_path){

    //model_path = path;

//    model_path = model_path;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    auto env_ = std::make_unique<Ort::Env>( ORT_LOGGING_LEVEL_WARNING, "experiment" );
    fdet_env = std::move(env_);

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//
    printf("Using Onnxruntime C++ API\n");
//    Ort::Session session(env, model_path, session_options);
    auto session = std::make_unique<Ort::Session>(*fdet_env.get(), det_path, session_options);
    fdet_session = std::move(session);

    auto meminf = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    det_memory_info = std::move(meminf);

//    fdet_session = & session;
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

//    input_name = session.GetInputName(0, allocator);
    det_input_node_names.push_back( fdet_session->GetInputName(0, allocator) );
    Ort::TypeInfo in_type_info = fdet_session->GetInputTypeInfo(0);
    auto tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    det_input_node_dims = tensor_info.GetShape();
    det_input_node_dims.at(2) = det_inHeight;
    det_input_node_dims.at(3) = det_inWidth;

    numOutputNodes = fdet_session->GetOutputCount();
    auto output_info = fdet_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    det_output_node_dims.push_back( output_info.at(1) * output_info.at(2) );

    output_info = fdet_session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    det_output_node_dims.push_back( output_info.at(1) * output_info.at(2) );

    output_info = fdet_session->GetOutputTypeInfo(2).GetTensorTypeAndShapeInfo().GetShape();
    det_output_node_dims.push_back( output_info.at(1) * output_info.at(2) );

    det_output_node_names.push_back( fdet_session->GetOutputName(0, allocator) );
    det_output_node_names.push_back( fdet_session->GetOutputName(1, allocator) );
    det_output_node_names.push_back( fdet_session->GetOutputName(2, allocator) );

    calculateAnchors();

    // feature extraction part
    auto asession = std::make_unique<Ort::Session>(*fdet_env.get(), align_path, session_options);
    feat_session = std::move(asession);
    auto ameminf = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    feat_memory_info = std::move(ameminf);
    Ort::AllocatorWithDefaultOptions aallocator;
    //    input_name = session.GetInputName(0, allocator);
    feat_input_node_names.push_back( feat_session->GetInputName(0, aallocator) );
    Ort::TypeInfo ain_type_info = feat_session->GetInputTypeInfo(0);
    auto atensor_info = ain_type_info.GetTensorTypeAndShapeInfo();
    type = atensor_info.GetElementType();
    feat_input_node_dims = atensor_info.GetShape();

    // get output infor
    auto out_type_info = feat_session->GetOutputTypeInfo(0);
    auto aoutput_info = out_type_info.GetTensorTypeAndShapeInfo();

    feat_output_node_dims = aoutput_info.GetShape();
    feat_output_node_dims.at(0) = 1;
    feat_input_node_dims.at(2) = feat_inHeight;
    feat_input_node_dims.at(3) = feat_inWidth;

    feat_output_node_names.push_back( feat_session->GetOutputName(0, aallocator) );


}


void faceprocess::do_forward() {

    // setting input info

    det_input_node_dims.at(0) = det_batch_size;

    det_inputTensors.clear();
    detectedFaces.clear();


    // in case of multiple tensors ---
    det_inputTensors.push_back( Ort::Value::CreateTensor<float>(
            *det_memory_info.get(), det_inputTensorValues.data(), det_input_tensor_size, det_input_node_dims.data(), det_input_node_dims.size() ) );

    // score model & input tensor, get back output tensor
    // small detector
    auto output_tensors = fdet_session.get()->Run(Ort::RunOptions{nullptr}, det_input_node_names.data(), det_inputTensors.data(), det_input_node_names.size(), det_output_node_names.data(), det_output_node_names.size());

    // decode boxes and landmarks
    std::vector<FaceInfo> decodedFaces = decode_bboxes(output_tensors[0].GetTensorMutableData<float>(), output_tensors[1].GetTensorMutableData<float>(),
                  output_tensors[2].GetTensorMutableData<float>());

    // apply nms
    tiny_nms( decodedFaces );

    // finally sorting face according to batch order
    // Very important in batch processing
    std::sort(detectedFaces.begin(), detectedFaces.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.batchOrder < b.batchOrder; });


}

void faceprocess::get_features(){
    // landmark reg model
    feat_input_node_dims.at(0) = feat_batch_size;
    feat_input_tensor_size = 1;
    for (int i=0; i<feat_input_node_dims.size();i++)
        feat_input_tensor_size *= feat_input_node_dims.at(i);

    feat_inputTensors.clear();

    feat_inputTensors.push_back( Ort::Value::CreateTensor<float>(
            *feat_memory_info.get(), feat_inputTensorValues.data(), feat_input_tensor_size, feat_input_node_dims.data(), feat_input_node_dims.size() ) );

    // score model & input tensor, get back output tensor
    auto output_tensors = feat_session.get()->Run(Ort::RunOptions{nullptr}, feat_input_node_names.data(), feat_inputTensors.data(), feat_input_node_names.size(), feat_output_node_names.data(), feat_output_node_names.size());

    // ---< filling out the features  >
    float* total_features = output_tensors[0].GetTensorMutableData<float>();
    for (int i=0; i<detectedFaces.size(); i++) {
        // using memcpy
        memcpy( &detectedFaces[i].features[0], &total_features[i * feature_size], feature_size*sizeof (total_features[0]));
    }

}


std::vector<cv::Mat> faceprocess::set_input(std::vector<cv::Mat> inputImg, char * task) {
    //inputImg = preprocess_img(inputImg, inWidth, inHeight);
    std::vector<cv::Mat> processed_img;

    if (std::strcmp( task, "det"  ) == 0 ) {
        //float det_inputValues[13* det_inWidth * det_inHeight*3];
        std::vector<float> det_inputValues(inputImg.size() * det_inWidth * det_inHeight * 3);
        processed_img = imgsToFlattenedVector(inputImg, det_inputValues.data());
        det_inputTensorValues.clear(); // copy data from one vector to another
        copy(det_inputValues.begin(), det_inputValues.end(), back_inserter(det_inputTensorValues));
    }else{
        // old algorithm
        std::vector<float> feat_inputValues(inputImg.size() * feat_inWidth * feat_inHeight * 3);
        batchOptimizedWrappFlatten(det_inputTensorValues, feat_inputValues);
        feat_inputTensorValues.clear();
        copy(feat_inputValues.begin(), feat_inputValues.end(), back_inserter(feat_inputTensorValues));
    }

    return processed_img;
}

// batch optimized wrapping img -----------
int faceprocess::batchOptimizedWrappFlatten( std::vector<float>&srcs, std::vector<float>& wrappedFlattenedFaces){
    // the code shoul be modified
    // get transformation matri
    // face alignment should be added here!
    // old single batch process ---
    fp_t transformation_mat[2][3];
    // iterating over images ---
    int singleJump = feat_inHeight * feat_inWidth;
    int jumpImg = det_inWidth * det_inHeight;
    feat_batch_size =1;
    float landmarks[10];
    std::vector<unsigned char> tmpface;

    for (int bi =0; bi<detectedFaces.size(); bi++) {
        memcpy(&landmarks, detectedFaces[bi].landmark5, 10*sizeof(detectedFaces[bi].landmark5[0]) );
        bool trfstat = transformation_matrix( landmarks , transformation_mat);
        //if (! trfstat)
        //    continue;
        // get source data -> hard coded wrapping goes here --- not finished yet

        dt_p rindex[2];
        //
        dt_p b1234[4] = {0, 0, 0, 0};

        //
        for (int i = 0; i < feat_inHeight; i++) {
            // go to the next line
            for (int j = 0; j < feat_inWidth; j++) {
                // aligment and allocation at the same time
                invert_indx(j, i, rindex, transformation_mat);
                // get neigbours
                int32_t neighbour[4] = {0, 0, 0, 0};

                if (std::floor(rindex[0]) >= 0 && std::ceil(rindex[0]) < det_inHeight && std::floor(rindex[1]) >= 0 &&
                    std::ceil(rindex[1]) < det_inWidth) {
                    neighbour[0] = std::floor(rindex[0]);
                    neighbour[1] = std::ceil(rindex[0]);
                    neighbour[2] = std::floor(rindex[1]);
                    neighbour[3] = std::ceil(rindex[1]);

                } else {

                    if (rindex[0] < 0) {
                        neighbour[0] = 0;
                        neighbour[1] = 1;
                    } else if (rindex[0] >= det_inHeight) {
                        neighbour[0] = det_inHeight - 2;
                        neighbour[1] = det_inHeight - 1;
                    }

                    if (rindex[1] <= 0) {
                        neighbour[2] = 0;
                        neighbour[3] = 1;
                    } else if (rindex[1] >= det_inWidth) {
                        neighbour[2] = det_inWidth - 2;
                        neighbour[3] = det_inWidth - 1;
                    }

                }
                // to modify this
                for (int c = 0; c < 3; c++) // dst-- for RRGGBB version - inverting flattened indexes from detection and wrapping them out
                {
                    b1234[0] = *(srcs.data() + (neighbour[0] * det_inWidth + neighbour[2])  + c * jumpImg + bi * 3 * jumpImg  ) + det_mean_vals[c]; // 3 * (neighbour[0] * srcWidth + neighbour[2]) + c
                    b1234[1] = *(srcs.data() + (neighbour[1] * det_inWidth + neighbour[2]) + c* jumpImg + bi * 3 * jumpImg ) -
                               *(srcs.data() + (det_inWidth * neighbour[0] + neighbour[2]) + c* jumpImg + bi * 3 * jumpImg ); // *(src + 3 * (neighbour[1] * det_inWidth + neighbour[2]) + c) - *(src + 3 * (det_inWidth * neighbour[0] + neighbour[2]) + c);
                    b1234[2] = *(srcs.data() + (neighbour[0] * det_inWidth + neighbour[3]) + c* jumpImg + bi * 3 * jumpImg) -
                               *(srcs.data() + (det_inWidth * neighbour[0] + neighbour[2]) + c* jumpImg + bi * 3 * jumpImg);
                    b1234[3] = *(srcs.data() +  (neighbour[0] * det_inWidth + neighbour[2]) + c* jumpImg + bi * 3 * jumpImg) -
                               *(srcs.data() + (det_inWidth * neighbour[1] + neighbour[2]) + c* jumpImg + bi * 3 * jumpImg) -
                               *(srcs.data() + (neighbour[0] * det_inWidth + neighbour[3]) + c* jumpImg + bi * 3 * jumpImg) +
                               *(srcs.data() + (det_inWidth * neighbour[1] + neighbour[3]) + c* jumpImg + bi * 3 * jumpImg) ;
                    //
                    int32_t intensity = b1234[0] + b1234[1] * (rindex[0] - neighbour[0]) * (neighbour[3] - rindex[1]) +
                                        b1234[2] * (rindex[1] - neighbour[2]) * (neighbour[1] - rindex[0]) +
                                        b1234[3] * (rindex[0] - neighbour[0]) * (rindex[1] - neighbour[2]);

                    intensity = (intensity<=255) ? intensity : 255;
                    intensity = ( intensity>=0) ? intensity : 0;

                    // todo: this is still HWC
                    wrappedFlattenedFaces.at(i * feat_inHeight + j + c* singleJump + bi * 3 * singleJump) =  (float)(feat_inScaleFactor[0]*intensity-feat_inScaleFactor[1]);
                }
            }

        }; // end of the single alignment


        // write down to check out the image
        //todo: this is for test
        // tmpface.push_back(intensity);

        tmpface.clear();
        copy(wrappedFlattenedFaces.data() + bi * 3 * singleJump, wrappedFlattenedFaces.data()+ (bi+1) * 3 * singleJump, back_inserter(tmpface));
        std::ofstream ofs;
        std::string imgPath = "face_" + std::to_string(bi);
        ofs.open( imgPath, std::ios::binary);

        ofs << "P6" << "\n";
        ofs << 112 << " ";
        ofs << 112 << "\n";
        ofs << 255 << "\n";
        ofs.write((char *) tmpface.data(), 112 * 112 * 3);

        ofs.clear();
        ofs.close();


    }// end of loop over the images in batch

    return 1;

}



// Retina face tools

std::vector<FaceInfo> faceprocess::decode_bboxes(float *bboxes, float *confidence, float *landmarks ) {

    FaceInfo facecandid;
    std::vector<FaceInfo> facecandids;
    float box[4];
    //float landmark[10];
    //std::vector<std::vector<float>> decoded_bboxs;
    int startIndex = 0, sind;

    for (int j=0; j< det_batch_size; j++) {

        for (int i = 0; i < anchors.size(); i++) {

            sind = i+startIndex;
            if (confidence[2 * sind + 1] < score_threshold)
                continue;

            box[0] = ((anchors.at(i)[0] + bboxes[4 * sind] * variance[0] * anchors.at(i)[2]) * det_inWidth);
            box[1] = ((anchors.at(i)[1] + bboxes[4 * sind + 1] * variance[0] * anchors.at(i)[3]) * det_inHeight);
            box[2] = (anchors.at(i)[2] * std::exp(bboxes[4 * sind + 2] * variance[1]) * det_inWidth);
            box[3] = (anchors.at(i)[3] * std::exp(bboxes[4 * sind + 3] * variance[1]) * det_inHeight);

            box[0] -= box[2] / 2;
            box[1] -= box[3] / 2;
            box[2] += box[0];
            box[3] += box[1];

            facecandid.x1 = box[0];
            facecandid.y1 = box[1];
            facecandid.x2 = box[2];
            facecandid.y2 = box[3];

            facecandid.score = confidence[2 * sind + 1];

            // landmark decode
            for (int k = 0; k < 5; k++) {
                facecandid.landmark5[2 * k] = (
                        (anchors.at(i)[0] + landmarks[10 * sind + 2 * k] * variance[0] * anchors.at(i)[2]) * det_inWidth);
                facecandid.landmark5[2 * k + 1] = (
                        (anchors.at(i)[1] + landmarks[10 * sind + 2 * k + 1] * variance[0] * anchors.at(i)[3]) *
                        det_inHeight);
            }
            facecandid.batchOrder = j;
            facecandids.push_back(facecandid);
        }

        startIndex +=anchors.size();
    }

    return facecandids;
};




void faceprocess::tiny_nms(std::vector<FaceInfo> input) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold && input[i].batchOrder == input[j].batchOrder) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        detectedFaces.push_back(buf[0]);

    }
}

// feature
bool transformation_matrix( fp_t flandmarks[10],fp_t M[2][3]){

    // calculate mean and variance of 5 landmarks
    dt_p fl_mean[2]; dt_p fl_var;
    fl_mean[0] = (flandmarks[0] + flandmarks[2] + flandmarks[4] + flandmarks[6] + flandmarks[8])/5.;
    fl_mean[1] = (flandmarks[1] + flandmarks[3] + flandmarks[5] + flandmarks[7] + flandmarks[9])/5.;
    // normalization
    dt_p mmx=0, mmy=0;
    for (int i=0; i<5; i++){
        flandmarks[2*i] -= fl_mean[0];
        flandmarks[2*i+1] -= fl_mean[1];

        mmx += flandmarks[2*i]/5;
        mmy += flandmarks[2*i+1]/5;
    }
    fl_var = (pow(flandmarks[0] - mmx,2) + pow(flandmarks[1] - mmy,2) + pow(flandmarks[2] - mmx,2)
              + pow(flandmarks[3] - mmy,2) + pow(flandmarks[4] - mmx,2) + pow(flandmarks[5] - mmy,2) + pow(flandmarks[6] - mmx,2)
              + pow(flandmarks[7] - mmy,2) + pow(flandmarks[8] - mmx,2) + pow(flandmarks[9] - mmy,2)) /5 ;

    // matrix multiplication part (srd[2x5] x dst[5x2])
    dt_p matrixpxp [4] = {0,0,0,0};
    for (int i=0; i<5;i++) {
        matrixpxp[0] += (1 / 5.) * flandmarks[2*i] * reference_landmark.standardized[i][0];
        matrixpxp[1] += (1. / 5.) * flandmarks[2*i+1] * reference_landmark.standardized[i][0];
        matrixpxp[2] += (1. / 5.) * flandmarks[2*i] * reference_landmark.standardized[i][1];
        matrixpxp[3] += (1. / 5.) * flandmarks[2*i+1] * reference_landmark.standardized[i][1];
    }
    // compute determinant of A
    dt_p detA = matrixpxp[0] * matrixpxp[3] - matrixpxp[2] * matrixpxp[1];
    dt_p d[2] = {1,1};
    int rankA;
    if (detA<0)
        d[1] = -1;
    if (detA < SMALL_NUM || -detA > -1*SMALL_NUM)
        rankA = (matrixpxp[0] == 0 || matrixpxp[1]==0 || matrixpxp[2]==0 || matrixpxp[3]==0) ? 0:1;
    else
        rankA = 2;

    if (rankA==0)
        return false;

    // computing svd
    //a, b, c ,d = Mat[0], Mat[1], Mat[3], Mat[4]
    dt_p theta = .5 * std::atan2( 2*matrixpxp[0]*matrixpxp[2] + 2*matrixpxp[1]*matrixpxp[3], pow(matrixpxp[0],2) + pow(matrixpxp[1],2) - pow(matrixpxp[2],2) - pow(matrixpxp[3],2));
    dt_p psi = .5 * std::atan2(2*matrixpxp[0]*matrixpxp[1] + 2*matrixpxp[2]*matrixpxp[3], pow(matrixpxp[0],2) - pow(matrixpxp[1],2) + pow(matrixpxp[2],2) - pow(matrixpxp[3],2));

    dt_p U[4];
    U[0] = cos(theta); U[1] = std::sin(theta), U[2] = std::sin(theta), U[3] = -std::cos(theta);

    dt_p S1 = pow(matrixpxp[0],2) + pow(matrixpxp[1],2) + pow(matrixpxp[2],2) + pow(matrixpxp[3],2);

    dt_p S2 = std::sqrt(pow(pow(matrixpxp[0],2) + pow(matrixpxp[1],2) - pow(matrixpxp[2],2) - pow(matrixpxp[3],2), 2) + 4*pow(matrixpxp[0]*matrixpxp[2] + matrixpxp[1]*matrixpxp[3], 2));
    //
    dt_p S[2];
    S[0] = std::sqrt((S1 + S2)/2);
    S[1] = std::sqrt((S1 - S2)/2);
//    S[0] = {sig1, sig2}

    dt_p s11 = (matrixpxp[0]*std::cos(theta) + matrixpxp[2]*std::sin(theta))*std::cos(psi) + (matrixpxp[1]*std::cos(theta) + matrixpxp[3]*std::sin(theta))*std::sin(psi);
    dt_p s22 = (matrixpxp[0]*std::sin(theta) + matrixpxp[2]*std::cos(theta))*std::sin(psi) + (-matrixpxp[1]*std::sin(theta) + matrixpxp[3]*std::cos(theta))*std::cos(psi);

    float Vt [4];
    Vt[0] = s11/std::abs(s11) * std::cos(psi); Vt[1] = s11/std::abs(s11) * std::sin(psi);
    Vt[2] = s22/std::abs(s22) * std::sin(psi); Vt[3] = -s22/std::abs(s22) * std::cos(psi);

    // conreolling the rank of the matrix
    // in case of low rank matrices
    if (rankA==1){
        dt_p detU = U[0]*U[3] - U[1]*U[2];
        dt_p detV = Vt[0]*U[3] - U[1]*U[2];
        if (detU * detV == 0){
            M[0][0] = U[0]*Vt[0] + U[1]*Vt[1];
            M[0][1] = U[0]*Vt[2] + U[1]*Vt[3];
            M[1][0] = U[2]*Vt[0] + U[3]*Vt[1];
            M[1][1] = U[2]*Vt[2] + U[3]*Vt[3];
            M[0][2] = 0;
            M[1][2] = 0;
        } else{
            uint8_t tmps = d[1];
            d[1] = -1;
            M[0][0] = d[0]*U[0]*Vt[0] + d[1]*U[1]*Vt[1];
            M[0][1] = d[0]*U[0]*Vt[2] + d[1]*U[1]*Vt[3];
            M[1][0] = d[0]*U[2]*Vt[0] + d[1]*U[3]*Vt[1];
            M[1][1] = d[0]*U[2]*Vt[2] + d[1]*U[3]*Vt[3];
            M[0][2] = 0;
            M[1][2] = 0;
            d[1] = tmps;
        }
    }else{
        M[0][0] = d[0]*U[0]*Vt[0] + d[1]*U[1]*Vt[1];
        M[0][1] = d[0]*U[0]*Vt[2] + d[1]*U[1]*Vt[3];
        M[1][0] = d[0]*U[2]*Vt[0] + d[1]*U[3]*Vt[1];
        M[1][1] = d[0]*U[2]*Vt[2] + d[1]*U[3]*Vt[3];
        M[0][2] = 0;
        M[1][2] = 0;
    }

    // scale is true
    //if (estimate_scale):
    // Eq. (41) and (42).
    // compute S x d
    dt_p scale = 1.0 / fl_var * (S[0]* d[0] + S[1]* d[1]);
    //
    M[0][2] = reference_landmark.ref_mean[0] - scale * (M[0][0] * fl_mean[0] + M[0][1] * fl_mean[1]);
    M[1][2] = reference_landmark.ref_mean[1] - scale * (M[1][0] * fl_mean[0] + M[1][1] * fl_mean[1]) ;
    //
    M[0][0] *= scale;
    M[0][1] *= scale;
    M[1][0] *= scale;
    M[1][1] *= scale;
    //
    //M[2][0] = 0; M[2][1] = 0; M[2][2] = 1.;
    return true;
};

bool invert_indx( int indi, int indj, dt_p rindex[2], fp_t M[2][3]){
    //
    // get inverse of 2d matrix
    dt_p iindex[2];
    iindex[0] = indi - M[0][2]; // translation
    iindex[1] = indj - M[1][2]; //
    dt_p det = 1/ (M[0][0]*M[1][1] - M[0][1]*M[1][0]);
    if (det < 1e-9){
        std::cout<< " determinant is close to zero \n";
        return false;}
    // inversion
    rindex[1] = det * (iindex[0]*M[1][1] - iindex[1]*M[0][1]);
    rindex[0] = det * (-iindex[0]*M[1][0] + iindex[1]*M[0][0]);

    return true;
}


void faceprocess::get_feature_map(int16_t INPUT_H, int16_t INPUT_W, int feature_maps[3][2]) {
    for (int i = 0; i<3; i++) {
        feature_maps[i][0] = std::ceil(INPUT_H / steps[i]);
        feature_maps[i][1] = std::ceil(INPUT_W / steps[i]);
    }
};

//  std::vector<std::vector<float>>& anchors
void faceprocess::calculateAnchors() {

    // get feature map
    int feature_maps[3][2];
    get_feature_map(det_inWidth, det_inHeight, feature_maps);
    //    std::vector<std::vector<float>> anchors;
    //    std::vector<float> landms;
    std::vector<fp_t> anch;
    for (int16_t k = 0; k<3; k++) {
        //
        int i = 0, j = 0, fcnt = 0;
        while (fcnt < feature_maps[k][0] * feature_maps[k][1]) {
            j = fcnt % feature_maps[k][0];
            i = fcnt / feature_maps[k][0];
            fcnt++;
            //
            fp_t s_kx_1 = min_sizes[k][0] / det_inWidth;
            fp_t s_ky_1 = min_sizes[k][0] / det_inHeight;
            fp_t s_kx_2 = min_sizes[k][1] / det_inWidth;
            fp_t s_ky_2 = min_sizes[k][1] / det_inHeight;
            fp_t dense_cx = (j + 0.5) * steps[k] / det_inWidth;
            fp_t dense_cy = (i + 0.5) * steps[k] / det_inHeight;

            anch.clear();
            anch.push_back((dense_cx<1 ? dense_cx : 1.0) >0 ? (dense_cx<1 ? dense_cx : 1.0) : 0);
            anch.push_back((dense_cy<1 ? dense_cy : 1.0)>0 ? (dense_cy<1 ? dense_cy : 1.0) : 0);

            anch.push_back((s_kx_1<1 ? s_kx_1 : 1.0)>0 ? (s_kx_1<1 ? s_kx_1 : 1.0) : 0);
            anch.push_back((s_ky_1<1 ? s_ky_1 : 1.0)>0 ? (s_ky_1<1 ? s_ky_1 : 1.0) : 0);
            //
            anchors.push_back(anch);
            anch.clear();
            anch.push_back((dense_cx<1 ? dense_cx : 1.0) >0 ? (dense_cx<1 ? dense_cx : 1.0) : 0);
            anch.push_back((dense_cy<1 ? dense_cy : 1.0)>0 ? (dense_cy<1 ? dense_cy : 1.0) : 0);
            anch.push_back((s_kx_2<1 ? s_kx_2 : 1.0)>0 ? (s_kx_2<1 ? s_kx_2 : 1.0) : 0);
            anch.push_back((s_ky_2<1 ? s_ky_2 : 1.0)>0 ? (s_ky_2<1 ? s_ky_2 : 1.0) : 0);
            anchors.push_back(anch);
            //                }
        }
    }

}