#include<cstdio>
#include<fstream>
#include<iostream>
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>

#include"tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#define INPUT_MEAN 128.0f
#define INPUT_STD 128.0f

using namespace tflite;

#define TFLITE_MODEL_CHECK(X)  \
if(!(X)){   \
    fprintf(stderr,"Error at %s:%d\n",__FILE__,__LINE__);  \
}

void MatFrameToTensor(cv::Mat frame,TfLiteTensor *input)
{
    const int size=input->dims->data[1]*input->dims->data[2]*input->dims->data[3];
    switch(input->type)
    {
        case kTfLiteFloat32:
        for(int i=0;i<size;i++)
        {
            input->data.f[i]=(frame.data[i]-INPUT_MEAN)/INPUT_STD;
        }
        break;
        case kTfLiteUInt8:
        memcpy(input->data.uint8,frame.data,size);
        break;
        default:
        std::cout<<"Should not reach here! \n";
    }
}

int main(int argc,char *argv[])
{
    const char *model_filename=argv[1];
    const char *video_filename=argv[2];
    const char *score_thres=argv[3];
    const float score_threshold=atof(score_thres);

    fprintf(stdout,"Reading model from %s\n",argv[1]);
    fprintf(stdout,"Reading video from %s\n",argv[2]);
    std::cout<<score_threshold<<std::endl;
    //fprintf(stdout,"Score_threshold is %f\n",&score_threshold);

    //load Model
    std::unique_ptr<tflite::FlatBufferModel>model=tflite::FlatBufferModel::BuildFromFile(model_filename);
    TFLITE_MODEL_CHECK(model!=nullptr);

    //build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model.get(),resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MODEL_CHECK(interpreter!=nullptr);

    //Allocate tensor buffers
    TFLITE_MODEL_CHECK(interpreter->AllocateTensors()==kTfLiteOk);

    //input size of tf model
    int input=interpreter->inputs()[0];
    const uint32_t input_width=interpreter->tensor(input)->dims->data[1];
    std::cout<<input_width<<std::endl;
    const uint32_t input_height=interpreter->tensor(input)->dims->data[2];
    std::cout<<input_height<<std::endl;

    //video capture using opencv
    cv::Mat frameImg,frameImg_tf;
    cv::VideoCapture capture;
    capture.open(video_filename);
    if(!capture.isOpened())
    {
        std::cout<<"Failed to open video file"<<std::endl;
        return -1;
    }

    //capture the frames using detection model
    while(true)
    {
        capture>>frameImg;
        if(!frameImg.empty())
        {
            cv::resize(frameImg,frameImg_tf,cv::Size(input_width,input_height));
            cv::cvtColor(frameImg_tf,frameImg_tf,CV_BGR2RGB);
            MatFrameToTensor(frameImg_tf,interpreter->tensor(input));

            //run inference
            if(interpreter->Invoke()!=kTfLiteOk)
            {
                std::cout<<"Error"<<std::endl;
                return false;
            }
            float* output_locations=interpreter->typed_output_tensor<float>(0);
            float* output_classes=interpreter->typed_output_tensor<float>(1);
            float* output_scores=interpreter->typed_output_tensor<float>(2);
            float* output_detections=interpreter->typed_output_tensor<float>(3);

            const char *class_label[]={"face","calling","smoking"};
            for(int d=0;d<*output_detections;d++)
            {
                //const std::string cls=labels[output_classes[d]];
                const std::string cls=class_label[int(output_classes[d])];
                const float score=output_scores[d];
                const int ymin=output_locations[4*d]*frameImg.rows;
                const int xmin=output_locations[4*d+1]*frameImg.cols;
                const int ymax=output_locations[4*d+2]*frameImg.rows;
                const int xmax=output_locations[4*d+3]*frameImg.cols;
                
                if(score<score_threshold)
                {
                    std::cout<<"Ignore detection "<<d<<" of "<<cls<<" detections with score "
                    <<score<<"@["<<xmin<<","<<ymin<<","<<xmax<<","<<ymax<<"]\n";
                }
                else
                {
                    std::cout<<"detected "<<d<<" of "<<cls<<"with score "<<score
                    <<"@["<<xmin<<","<<ymin<<","<<xmax<<","<<ymax<<"]"<<std::endl;
                    cv::rectangle(frameImg,cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin),
                    cv::Scalar(255,255,255),1);
                    cv::putText(frameImg,cls,cv::Point(xmin,ymin-5),
                    cv::FONT_HERSHEY_COMPLEX,.8,cv::Scalar(10,255,30));
                }
                cv::imshow("detected result",frameImg);

            }//end of for loop
        }//end of if(!frameImg.empty())
        else
        {
            break;
        }
        if(char(cv::waitKey(1))=='q')
        break;
    }//end of while loop


}//end of main