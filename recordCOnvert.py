tflite_convert --output_file=/tmp/tflite/test2.tflite --graph_def_file=/tmp/tflite/tflite_graph.pb --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --input_shapes=1,300,300,3 --allow_custom_ops --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=128 --default_ranges_min=0 --default_ranges_max=6


tflite_convert --output_file=/tmp/tflite/test.tflite --graph_def_file=/tmp/tflite/tflite_graph.pb --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --input_shapes=1,300,300,3 --allow_custom_ops


export CONFIG_FILE=PATH_TO_BE_CONFIGURED/pipeline.config
export CHECKPOINT_PATH=PATH_TO_BE_CONFIGURED/model.ckpt
export OUTPUT_DIR=/tmp/tflite

object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true