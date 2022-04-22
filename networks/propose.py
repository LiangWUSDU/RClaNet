from keras.models import Model
from keras.layers import Conv3D,Input, concatenate,Conv3DTranspose
from keras.layers import MaxPooling3D,Flatten,Dense,Lambda
from keras.initializers import RandomNormal
from functions.ext.neuron.layers import SpatialTransformer
import keras.layers as KL
from functions.voxmorphloss import multi_scale_loss
from networks.model_functions import myConv,encoder,decoder_SAM_multiscale,BatchActivate
def my_model(vol_size, enc_nf, dec_nf,mode = 'train',indexing='ij',src_feats=1, tgt_feats=1,position_feats = 3):

    src = Input(shape=[*vol_size, src_feats])
    tgt = Input(shape=[*vol_size, tgt_feats])
    position = Input(shape=[*vol_size, position_feats])
    con_path = concatenate([src,tgt],axis=-1)
    to_decoder_moving = encoder(con_path,enc_nf)
    Brider_path = myConv(to_decoder_moving[3], enc_nf[4], 2)
    decoder_path8,decoder_path16,decoder_path32,decoder_path64 = decoder_SAM_multiscale(Brider_path, to_decoder_moving,enc_nf,dec_nf)  #64,64,64
##  配准模块
    ##flow 为变形场，y为变形图像
    decoder_path8_up =Conv3DTranspose(dec_nf[0],kernel_size=(1,1,1), activation='relu', strides=(8,8,8),
                                      kernel_initializer='he_normal', use_bias=False)(decoder_path8)
    decoder_path8_up = concatenate([decoder_path8_up,position],axis=-1)
    decoder_path8_up = myConv(decoder_path8_up,dec_nf[0])
    flow8 = Conv3D(dec_nf[5], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow8')(decoder_path8_up)
    y8 = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_image8')([src, flow8])

    decoder_path16_up =Conv3DTranspose(dec_nf[1],kernel_size=(1,1,1), activation='relu', strides=(4,4,4),
                                      kernel_initializer='he_normal', use_bias=False)(decoder_path16)
    decoder_path16_up = concatenate([decoder_path16_up,position],axis=-1)
    decoder_path16_up = myConv(decoder_path16_up,dec_nf[1])
    flow16 = Conv3D(dec_nf[5], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow16')(decoder_path16_up)
    y16 = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_image16')([src, flow16])

    decoder_path32_up =Conv3DTranspose(dec_nf[2],kernel_size=(1,1,1), activation='relu', strides=(2,2,2),
                                      kernel_initializer='he_normal', use_bias=False)(decoder_path32)
    decoder_path32_up = concatenate([decoder_path32_up,position],axis=-1)
    decoder_path32_up = myConv(decoder_path32_up,dec_nf[2])
    flow32 = Conv3D(dec_nf[5], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow32')(decoder_path32_up)
    y32 = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_image32')([src, flow32])

    decoder_path64 = concatenate([decoder_path64,position],axis=-1)
    decoder_path64 = myConv(decoder_path64,dec_nf[3])
    flow = Conv3D(dec_nf[5], kernel_size=3, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5),name='flow' )(decoder_path64)
    y = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_image')([src, flow])
##分类模块1
    c1 = Conv3D(1, kernel_size=3, padding='same',activation='sigmoid',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='disease_risk')(decoder_path64)
    c11 =  Conv3D(dec_nf[6], kernel_size=3,padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c1)
    c11 = BatchActivate(c11)
    c11 = MaxPooling3D((2,2,2))(c11)
    c12 = Conv3D(dec_nf[7], kernel_size=3,padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c11)
    c12 = BatchActivate(c12)
    c12 = MaxPooling3D((2,2,2))(c12)
    c13 = Conv3D(dec_nf[7], kernel_size=3,padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c12)
    c13 = BatchActivate(c13)
    c13 = MaxPooling3D((2,2,2))(c13)
    c14 = Flatten()(c13)
    c15 = Dense(dec_nf[8])(c14)
    c15 = BatchActivate(c15,name='coarse_feature')
    coarse_predict = Dense(dec_nf[9],activation='softmax',name = 'coarse_predict')(c15)
##分类模块2
    c2 = concatenate([flow,c1],axis=-1)
    c2 = Conv3D(1, kernel_size=3, padding='same', activation='sigmoid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='fine_disease_risk')(c2)
    c21 = Conv3D(dec_nf[6], kernel_size=3, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c2)
    c21 = BatchActivate(c21)
    c21 = MaxPooling3D((2,2,2))(c21)
    c22 = Conv3D(dec_nf[7], kernel_size=3,padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c21)
    c22 = BatchActivate(c22)
    c22 = MaxPooling3D((2,2,2))(c22)
    c23 = Conv3D(dec_nf[7], kernel_size=3,padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(c22)
    c23 = BatchActivate(c23)
    c23 = MaxPooling3D((2,2,2))(c23)
    c24 = Flatten()(c23)
    c25 = Dense(dec_nf[8])(c24)
    c25 = BatchActivate(c25,name='fine_feature')
    cf_feature = concatenate([c15,c25],axis=-1)
    fine_predict = Dense(dec_nf[9],activation='softmax',name = 'fine_predict')(cf_feature)
    if mode == 'train':

        src_seg = Input(shape=[*vol_size, 4], name="src_seg")
        tgt_seg = Input(shape=[*vol_size, 4], name="tgt_seg")
        src_seg_boundary = Input(shape=[*vol_size, 4], name="src_seg_boundary")
        tgt_seg_boundary = Input(shape=[*vol_size, 4], name="tgt_seg_boundary")
        src_CSF = Lambda(lambda x: x[..., 1:2], name='src_CSF')(src_seg)
        tgt_CSF = Lambda(lambda x: x[..., 1:2], name='tgt_CSF')(tgt_seg)
        src_GM = Lambda(lambda x: x[..., 2:3], name='src_GM')(src_seg)
        tgt_GM = Lambda(lambda x: x[..., 2:3], name='tgt_GM')(tgt_seg)
        src_WM = Lambda(lambda x: x[..., 3:4], name='src_WM')(src_seg)
        tgt_WM = Lambda(lambda x: x[..., 3:4], name='tgt_WM')(tgt_seg)
        src_CSF_boundary = Lambda(lambda x: x[..., 1:2], name='src_CSF_boundary')(src_seg_boundary)
        tgt_CSF_boundary = Lambda(lambda x: x[..., 1:2], name='tgt_CSF_boundary')(tgt_seg_boundary)
        src_GM_boundary = Lambda(lambda x: x[..., 2:3], name='src_GM_boundary')(src_seg_boundary)
        tgt_GM_boundary = Lambda(lambda x: x[..., 2:3], name='tgt_GM_boundary')(tgt_seg_boundary)
        src_WM_boundary = Lambda(lambda x: x[..., 3:4], name='src_WM_boundary')(src_seg_boundary)
        tgt_WM_boundary = Lambda(lambda x: x[..., 3:4], name='tgt_WM_boundary')(tgt_seg_boundary)
        wseg = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_seg')([src_seg, flow])  ##不能用nearest
        wseg_CSF = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_CSF_seg')([src_CSF, flow])  ##不能用nearest
        wseg_GM = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_GM_seg')([src_GM, flow])  ##不能用nearest
        wseg_WM = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_WM_seg')([src_WM, flow])  ##不能用nearest
##边界
        wseg_boundary = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_boundary_seg')([src_seg_boundary, flow])
        wseg_CSF_boundary = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_CSF_boundary_seg')([src_CSF_boundary, flow])  ##不能用nearest
        wseg_GM_boundary = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_GM_boundary_seg')([src_GM_boundary, flow])  ##不能用nearest
        wseg_WM_boundary = SpatialTransformer(interp_method='linear', indexing=indexing,name='reg_WM_boundary_seg')([src_WM_boundary, flow])  ##不能用nearest
##
        my_seg_dice_loss = KL.Lambda(
            lambda x: multi_scale_loss(*x, loss_type='doubledice-ssim-mean', loss_scales=[0, 1, 2, 4, 8, 16]),
            name="seg_dice")([wseg, tgt_seg])
        my_seg_CSF_dice_loss = KL.Lambda(
			lambda x: multi_scale_loss(*x, loss_type='doubledice-ssim-mean', loss_scales=[0, 1, 2, 4, 8, 16]),
			name="seg_CSF_dice")([wseg_CSF, tgt_CSF])
        my_seg_GM_dice_loss = KL.Lambda(
            lambda x: multi_scale_loss(*x, loss_type='doubledice-ssim-mean', loss_scales=[0, 1, 2, 4, 8, 16]),
            name="seg_GM_dice")([wseg_GM, tgt_GM])
        my_seg_WM_dice_loss = KL.Lambda(
			lambda x: multi_scale_loss(*x, loss_type='doubledice-ssim-mean', loss_scales=[0, 1, 2, 4, 8, 16]),
			name="seg_WM_dice")([wseg_WM, tgt_WM])
##
        my_seg_boundary_dice_loss = KL.Lambda(
            lambda x: multi_scale_loss(*x, loss_type='ssim-mean-mse', loss_scales=[0, 1, 2, 4, 8, 16]),
            name="seg_boundary_dice")([wseg_boundary, tgt_seg_boundary])
        my_seg_CSF_boundary_dice_loss = KL.Lambda(
			lambda x: multi_scale_loss(*x, loss_type='ssim-mean-mse', loss_scales=[0, 1, 2, 4, 8, 16]),
			name="seg_CSF_boundary_dice")([wseg_CSF_boundary, tgt_CSF_boundary])
        my_seg_GM_boundary_dice_loss = KL.Lambda(
			lambda x: multi_scale_loss(*x, loss_type='ssim-mean-mse', loss_scales=[0, 1, 2, 4, 8, 16]),
			name="seg_GM_boundary_dice")([wseg_GM_boundary, tgt_GM_boundary])
        my_seg_WM_boundary_dice_loss = KL.Lambda(
			lambda x: multi_scale_loss(*x, loss_type='ssim-mean-mse', loss_scales=[0, 1, 2, 4, 8, 16]),
			name="seg_WM_boundary_dice")([wseg_WM_boundary, tgt_WM_boundary])
        inputs = [src, tgt,position, src_seg,tgt_seg,src_seg_boundary,tgt_seg_boundary]  # [input_img, target]
        outputs = [flow,y8,y16,y32,y,coarse_predict,fine_predict, my_seg_dice_loss,my_seg_CSF_dice_loss,my_seg_GM_dice_loss,my_seg_WM_dice_loss,my_seg_boundary_dice_loss,my_seg_CSF_boundary_dice_loss,my_seg_GM_boundary_dice_loss,my_seg_WM_boundary_dice_loss]  # ,regulariser_loss]#[output, my_complex_loss]
    else:
        # predict阶段，就不用计算loss了所以这里不加入loss层和metric层
        inputs = [src, tgt,position]  # input_img
        outputs = [flow,y8,y16,y32,y,fine_predict]

    model = Model(inputs=inputs, outputs=outputs)
    return model