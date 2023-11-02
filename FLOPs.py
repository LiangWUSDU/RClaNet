import tensorflow as tf
from networks.my_model import my_model_CNN_single
from networks.my_model import my_model_single,my_model_single11
from networks.VGG import VGG16
from networks.ResNet import Resnet
from networks.GoogleNet import Googlenet
import keras.backend as K

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


#model = my_model_CNN_single((160,192,160,1))
#model = VGG16((160,192,160,1))
#model = Resnet((160,192,160,1))
#model = Googlenet((160,192,160,1))
model = my_model_single11((160,192,160,1))
model.summary()
model1 = my_model_single((160,192,160,1))
model1.summary()


#model.summary()
#print(get_flops(model))
