import tensorflow as tf
from networks.Label_Reg import labelreg
from networks.voxelmorph import unet
import keras.backend as K
from networks.propose import my_model

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


#model = my_model(vol_size=((64,64,64)),enc_nf = [16,32,32,32,16],dec_nf= [32, 32, 32, 32, 16, 3, 16, 32,128,2], mode='train',  indexing='ij')
#model = labelreg(vol_size=((64,64,64)), nc =[32,64,128,256,512])
model = unet(vol_size=((64,64,64)),enc_nf = [16,32,32,32],dec_nf= [32,32,32,32,32,16,16,3])


model.summary()
print(get_flops(model))
