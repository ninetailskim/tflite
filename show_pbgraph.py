# import tensorflow as tf 
# import os 

# model_name = 'tflite_graph.pb' 
# def create_graph(): 
#     with tf.gfile.FastGFile(os.path.join( model_dir, model_name), 'rb') as f: 
#         graph_def = tf.GraphDef() 
#         graph_def.ParseFromString(f.read()) 
#         tf.import_graph_def(graph_def, name='model') 
        
# create_graph() 
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node] 
# for tensor_name in tensor_name_list: 
#     print(tensor_name,'\n')

from tensorflow.python.framework import tensor_util
from google.protobuf import text_format 
import tensorflow as tf 
from tensorflow.python.platform import gfile 
from tensorflow.python.framework import tensor_util 

GRAPH_PB_PATH = '../model_zoo/.pb' #path to your .pb file 
with tf.Session() as sess: 
    print("load graph") 
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f: 
        graph_def = tf.GraphDef() 
        # Note: one of the following two lines work if required libraries are available 
        #text_format.Merge(f.read(), graph_def) 
        graph_def.ParseFromString(f.read()) 
        tf.import_graph_def(graph_def, name='') 
        for i,n in enumerate(graph_def.node):
             print("Name of the node - %s" % n.name)