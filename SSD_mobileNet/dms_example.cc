#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <typeinfo>
#include <ctime>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)


#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include <opencv2/opencv.hpp>
#include <cyc_dms.h>
using namespace tflite;

#define INPUT_MEAN 128.0f
#define INPUT_STD 128.0f
clock_t time_1, time_2, time_3;


using namespace std;

int main(int argc, char* argv[])
{

//	const char* model_filename = argv[1];
//	const char* video_filename = argv[2];
//	const char* labels_filename = argv[3];

	

  	int lRet = 0;

  	AppContext lContext;
     
  	//*** process command line parameters ***
  	printf("\n**************************************************************\n"
         "** Omniviion ar0135 viulite -> dcu demo\n"
         "** Description:\n"
         "**  o Omniviion ov10635 (on VIULITE_0) expected as image input.\n"
         "**  o ISP only assembles full frames in DDR buffers using FDMA.\n"
         "**  o Resulting YUV422 1280x720 image is displayed live using DCU.\n"
         "**\n"
         "** Usage:\n"
         "**  o no cmd line parameters available.\n"
         "**\n"
         "**************************************************************\n\n");
#ifndef __STANDALONE__  
  	fflush(stdout);  
  	sleep(1);
#endif // ifndef __STANDALONE__
  
  	if(Prepare(lContext) == 0)
  	{
    	    if(Run(lContext) != 0)
    	    {
      		printf("Demo execution failed.\n");
      		lRet = -1;
    	    } // if Run() failed
  	} // if Prepare() ok
  	else
  	{
    	    printf("Demo failed in preparation phase.\n");
    	    lRet = -1;
  	}// else from if Prepare() ok
  
  	if(Cleanup(lContext) != 0)
  	{
    	    printf("Demo failed in cleanup phase.\n");
    	    lRet = -1;
  	} // if cleanup failed

  	return lRet;
} // main()



static int32_t Run(AppContext &arContext)
{  
  //*** Init DCU Output ***
#ifdef __STANDALONE__
  	io::FrameOutputDCU lDcuOutput(
                      WIDTH, 
                      HEIGHT, 
                      io::IO_DATA_DEPTH_08, 
                      CHNL_CNT,
                      DCU_BPP_YCbCr422);
#else // #ifdef __STANDALONE__
  	// setup Ctrl+C handler
  	if(SigintSetup() != SEQ_LIB_SUCCESS) 
  	{
    	    VDB_LOG_ERROR("Failed to register Ctrl+C signal handler.");
    	    return -1;
  	}
  
  	printf("Press Ctrl+C to terminate the demo.\n");
  
  	io::FrameOutputV234Fb lDcuOutput(
                          1024, 
                          600, 
                          io::IO_DATA_DEPTH_08, 
                          CHNL_CNT,
                          DCU_BPP_24);
#endif // else from #ifdef __STANDALONE__
  
  	unsigned long lTimeStart = 0, lTimeEnd = 0, lTimeDiff = 0;
  
  	// *** start grabbing ***
  	GETTIME(&lTimeStart);
  	if(arContext.mpGrabber->Start() != LIB_SUCCESS)
  	{
	    printf("Failed to start the grabber.\n");
	    return -1;
  	} // if Start() failed
  
  	SDI_Frame lFrame;

	int i = 0;
	char buf[20];
	cv::Mat showMat;
	cv::Mat frameImg_tf;
	std::unique_ptr<FlatBufferModel> model=FlatBufferModel::BuildFromFile("detect_float.tflite");

	std::unique_ptr<Interpreter> interpreter;
	tflite::ops::builtin::BuiltinOpResolver resolver;
	InterpreterBuilder builder(*model, resolver);
	builder(&interpreter);
	interpreter->AllocateTensors();
	int input = interpreter->inputs()[0];

  	// *** grabbing/processing loop ***  
  	for(;;)
  	{
    	    lFrame = arContext.mpGrabber->FramePop();
    	    if(lFrame.mUMat.empty())
    	    {
	        printf("Failed to grab image number %u\n", arContext.mFrmCnt);
                arContext.mError = true;
	        break;
    	    } // if pop failed

    	    arContext.mFrmCnt++;
    	    cv::Mat mat = lFrame.mUMat.getMat(vsdk::ACCESS_RW | OAL_USAGE_CACHED);		//The algorithm uses this mat

	    /*Add algorithm processing*/
	    //main_dms(mat);
            //printf("\nfanghongyu:main_dms(mat)");
	    /*save the picture (.png)*/
			cv::resize(mat, frameImg_tf, cv::Size(300, 300));
			cv::cvtColor(frameImg_tf, frameImg_tf, CV_BGR2RGB);
			const int size = interpreter->tensor(input)->dims->data[1]*interpreter->tensor(input)->dims->data[2]*interpreter->tensor(input)->dims->data[3];
			std::vector<int> array_int(size);
			std::vector<float> array_float(size);
			for(int i=0; i<size; i++){
				// array[i] = ((frameImg_tf.data[i]) - INPUT_MEAN)/INPUT_STD;
				array_int[i] = frameImg_tf.data[i];
				array_float[i] = (float(array_int[i])-INPUT_MEAN)/INPUT_STD;
				interpreter->tensor(input)->data.f[i] = array_float[i];
			// 	interpreter->tensor(input)->data.f[i] = ((frameImg_tf.data[i]) - INPUT_MEAN)/INPUT_STD;
			}
			time_2 = clock();	
			interpreter->Invoke();
			float *output_locations = interpreter->typed_output_tensor<float>(0);
			float *output_classes = interpreter->typed_output_tensor<float>(1);
			float *output_scores = interpreter->typed_output_tensor<float>(2);
			float *output_detections = interpreter->typed_output_tensor<float>(3);
			time_3 = clock();
			for(int d=0; d<*output_detections; d++)
			{
				const int cls = output_classes[d]; 
				const float score = output_scores[d];
			//	const int y_min = output_locations[4*d]*frameImg.rows;
			//	const int x_min = output_locations[4*d+1]*frameImg.cols;
			//	const int y_max = output_locations[4*d+2]*frameImg.rows;
			//	const int x_max = output_locations[4*d+3]*frameImg.cols;

				if(score > 0.38){
					// cv::rectangle(frameImg, cv::Rect(x_min, y_min, x_max-x_min, y_max-y_min), cv::Scalar(255,255,255), 1);
					// cv::putText(frameImg, cls, cv::Point(x_min, y_min-5), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(10,255,30));
					//std::cout << "class:" << cls << "; "<< double(time_3-time_1)/CLOCKS_PER_SEC << "; "<<double(time_3-time_2)/CLOCKS_PER_SEC << "; "<<double(time_2-time_1)/CLOCKS_PER_SEC << std::endl;
					std::cout << cls << std::endl;
				}
			}
			// cv::imshow("hello", frameImg);
		

		// if(char(cv::waitKey(10))=='q') break;

    	    if(arContext.mpGrabber->FramePush(lFrame) != LIB_SUCCESS)
    	    {
		 printf("Failed to push image number %u\n", arContext.mFrmCnt);
		 arContext.mError = true;
		 break;
    	    } // if push failed    
    
    	    if((arContext.mFrmCnt % FRM_TIME_MSR) == 0)
    	    {
		 GETTIME(&lTimeEnd);
		 lTimeDiff  = lTimeEnd - lTimeStart;
		 lTimeStart = lTimeEnd;
      
      		 printf("%u frames took %lu usec (%5.2ffps)\n",
				 FRM_TIME_MSR,
				 lTimeDiff,
				 (FRM_TIME_MSR*1000000.0)/((float)lTimeDiff));
    	    }// if time should be measured
    
#ifndef __STANDALONE__  
	    if(sStop)
	    {
	  	break; // break if Ctrl+C pressed
	    } // if Ctrl+C
#endif //#ifndef __STANDALONE__  
  	} // for ever
    
  return 0;
} // Run()