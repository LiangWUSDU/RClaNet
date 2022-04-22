from functions.data_gen import *
import nibabel as nib
from functions.utils.patch_operations import slice_3Dmatrix
from scipy import ndimage

def location_generator():

    x = np.linspace(0, 160, 160, endpoint=False)
    y = np.linspace(0, 192, 192, endpoint=False)
    z = np.linspace(0, 160, 160, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, sparse=False, indexing='ij')
    X = bn(X)
    Y = bn(Y)
    Z = bn(Z)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    Z = Z[..., np.newaxis]
    location = np.concatenate([X, Y, Z], axis=-1)
    return location
def generator_data(train_moving_image_file, st,train_moving_mask_file,sl,train_moving_boundary_file):
    vol_dir = train_moving_image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_data()
    affine0 = image1.affine.copy()
    moving_image = np.asarray(image, dtype=np.float32)
    mask_dir = train_moving_mask_file+sl
    moving_mask = nib.load(mask_dir).get_data()
    moving_mask = np.asarray(moving_mask, dtype=np.float32)
    boundary_dir = train_moving_boundary_file+sl
    moving_boundary = nib.load(boundary_dir).get_data()
    moving_boundary = np.asarray(moving_boundary,dtype=np.float32)
    fixed_image = nib.load('train_data/train/fixed_image/MNI_T1_1mm.nii.gz').get_data()
    fixed_image = np.asarray(fixed_image, dtype=np.float32)
    fixed_mask = nib.load('train_data/train/fixed_mask/MNI_T1_1mm_seg.nii.gz').get_data()
    fixed_mask = np.asarray(fixed_mask, dtype=np.float32)
    fixed_boundary = nib.load('train_data/train/fixed_mask_boundary/MNI_T1_1mm_seg.nii.gz').get_data()
    fixed_boundary = np.asarray(fixed_boundary, dtype=np.float32)
    location = nib.load('train_data/train/location/location.nii.gz').get_data()
    location = np.asarray(location,dtype=np.float32)
    return moving_image,moving_mask,moving_boundary,fixed_image,fixed_mask,fixed_boundary,location,affine0

def generator_test_data(train_moving_image_file, st,train_moving_mask_file,sl):
    vol_dir = train_moving_image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_data()
    #affine = image1.affine.copy()
    #hdr = image1.header.copy()
    moving_image = np.asarray(image, dtype=np.float32)
    mask_dir = train_moving_mask_file+sl
    moving_mask = nib.load(mask_dir).get_data()
    moving_mask = np.asarray(moving_mask, dtype=np.float32)
    fixed_image1 = nib.load('train_data/test/fixed_image/MNI_T1_1mm.nii.gz')
    fixed_image = fixed_image1.get_data()
    affine = fixed_image1.affine.copy()
    hdr = fixed_image1.header.copy()
    fixed_image = np.asarray(fixed_image, dtype=np.float32)
    fixed_mask = nib.load('train_data/test/fixed_mask/MNI_T1_1mm_seg.nii.gz').get_data()
    fixed_mask = np.asarray(fixed_mask, dtype=np.float32)
    location = nib.load('train_data/test/location/location.nii.gz').get_data()
    location = np.asarray(location,dtype=np.float32)
    return moving_image,moving_mask,fixed_image,fixed_mask,location,affine,hdr
def bn(image):
	[x,y,z,h] = np.shape(image)
	image = np.reshape(image,(x*y*z*h,1))
	max = np.max(image)
	min = np.min(image)
	image = (image-min)/(max-min)
	image = np.reshape(image,(x,y,z,h))
	return image
def generator_patch(moving_image, moving_mask, fixed_image, fixed_mask,location):
    moving_image = moving_image[np.newaxis,...]
    moving_mask = moving_mask[np.newaxis,...]
    fixed_image = fixed_image[np.newaxis,...]
    fixed_mask = fixed_mask[np.newaxis,...]
    location = location[np.newaxis,...]
    moving_image_patch = vols_generator_patch(vol_name=moving_image, num_data=1, patch_size=[64,64,64],
                                                                     stride_patch=[32,32,32], out=1, num_images=80)
    fixed_image_patch = vols_generator_patch(vol_name=fixed_image, num_data=1, patch_size=[64,64,64],
                                             stride_patch=[32,32,32], out=1, num_images=80)
    moving_mask_patch = vols_mask_generator_patch(vol_name=moving_mask, num_data=1, patch_size=[64,64,64,4],
                                                  stride_patch=[32,32,32,4], out=1, num_images=80)
    fixed_mask_patch = vols_mask_generator_patch(vol_name=fixed_mask, num_data=1, patch_size=[64,64,64,4],
                                                 stride_patch=[32,32,32,4], out=1, num_images=80)
    location_patch = vols_location_generator_patch(vol_name=location, num_data=1, patch_size=[64,64,64,3],
                                                 stride_patch=[32,32,32,3], out=1, num_images=80)
    return moving_image_patch,moving_mask_patch,fixed_image_patch,fixed_mask_patch,location_patch

def gen_patch(moving_image, moving_mask, fixed_image, fixed_mask,location,moving_boundary,fixed_boundary):
     moving_image_patch = slice_3Dmatrix(moving_image, window=(64, 64, 64), overlap=(32,32,32))
     moving_mask_patch = slice_3Dmatrix(moving_mask, window=(64, 64, 64), overlap=(32,32,32))
     fixed_image_patch = slice_3Dmatrix(fixed_image, window=(64, 64, 64), overlap=(32,32,32))
     fixed_mask_patch = slice_3Dmatrix(fixed_mask, window=(64, 64, 64), overlap=(32,32,32))
     location_patch = slice_3Dmatrix(location, window=(64, 64, 64), overlap=(32,32,32))
     moving_boundary_patch = slice_3Dmatrix(moving_boundary, window=(64, 64, 64), overlap=(32, 32, 32))
     fixed_boundary_patch = slice_3Dmatrix(fixed_boundary, window=(64, 64, 64), overlap=(32, 32, 32))
     return moving_image_patch,moving_mask_patch,fixed_image_patch,fixed_mask_patch,location_patch,moving_boundary_patch,fixed_boundary_patch

def Get_Jac(displacement):
    displacement = np.reshape(displacement,(1,160,192,160,3))
    '''
    the expected input: displacement of shape(batch, H, W, D, channel),
    obtained in TensorFlow.
    '''
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 1])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_z[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    D = D1 - D2 + D3

    return D
def Random(moving_strings,moving_seg_strings,y_CDR):
    permutation = np.random.permutation(y_CDR.shape[0])
    moving_strings = np.array(moving_strings)[permutation]  # 训练数据
    moving_seg_strings = np.array(moving_seg_strings)[permutation]
    y_CDR = y_CDR[permutation]
    return moving_strings,moving_seg_strings,y_CDR
def Random_image(moving_image_patch,moving_mask_patch,fixed_image_patch,fixed_mask_patch,moving_image_patch_position,moving_boundary_patch,fixed_boundary_patch):
    permutation = np.random.permutation(len(moving_image_patch))
    moving_image_patch = np.array(moving_image_patch)[permutation] #训练数据
    moving_mask_patch = np.array(moving_mask_patch)[permutation] #训练数据
    fixed_image_patch = np.array(fixed_image_patch)[permutation] #训练数据
    fixed_mask_patch = np.array(fixed_mask_patch)[permutation] #训练数据
    moving_image_patch_position = np.array(moving_image_patch_position)[permutation] #训练数据
    moving_boundary_patch = np.array(moving_boundary_patch)[permutation] #训练数据
    fixed_boundary_patch = np.array(fixed_boundary_patch)[permutation] #训练数据
    return moving_image_patch,moving_mask_patch,fixed_image_patch,fixed_mask_patch,moving_image_patch_position,moving_boundary_patch,fixed_boundary_patch

def propose_gen(train_moving_image_file,train_moving_image_txt,train_moving_mask_file,train_moving_mask_txt,train_moving_boundary_file,y_status,batch_size,batch_patch_size):
    while True:
        flow = np.zeros((1,64,64,64,3))
        y_true = np.zeros((1,1))
        image_file = open(train_moving_image_txt)  # 训练数据的名字放到txt文件里
        image_strings = image_file.readlines()
        mask_file = open(train_moving_mask_txt)
        mask_strings = mask_file.readlines()
        image_strings,mask_strings,y_CDR = Random(image_strings,mask_strings,y_status)
        for start in range(0, len(image_strings), batch_size):
            end = min(start + batch_size, len(image_strings))
            for id in range(start, end):
                st = image_strings[id].strip()  # 文件名
                sl = mask_strings[id].strip()
                y_label = y_CDR[id]
                y_label = np.reshape(y_label, (1, 2))
                moving_image, moving_mask,moving_boundary, fixed_image, fixed_mask,fixed_boundary, location,affine0 = generator_data(train_moving_image_file, st,train_moving_mask_file,sl,train_moving_boundary_file)
                moving_mask = to_categorical(moving_mask)
                fixed_mask = to_categorical(fixed_mask)
                moving_image_patch, moving_mask_patch, fixed_image_patch, fixed_mask_patch,moving_image_patch_position,moving_boundary_patch,fixed_boundary_patch = gen_patch( moving_image, moving_mask, fixed_image, fixed_mask,location,moving_boundary,fixed_boundary)
                moving_image_patch, moving_mask_patch, fixed_image_patch, fixed_mask_patch,moving_image_patch_position,moving_boundary_patch,fixed_boundary_patch = Random_image(moving_image_patch,moving_mask_patch,fixed_image_patch,fixed_mask_patch,moving_image_patch_position,moving_boundary_patch,fixed_boundary_patch)
                for start in range(0, len(moving_image_patch), batch_patch_size):
                    moving_image_batch_patch = []
                    fixed_image_batch_patch = []
                    moving_mask_batch_patch = []
                    fixed_mask_batch_patch = []
                    position = []
                    moving_boundary_batch_patch = []
                    fixed_boundary_batch_patch = []
                    end = min(start + batch_patch_size, len(moving_image_patch))
                    for id in range(start, end):
                        moving_image_patch1 = moving_image_patch[id]
                        fixed_image_patch1 = fixed_image_patch[id]
                        moving_mask_patch1 = moving_mask_patch[id]
                        fixed_mask_patch1 = fixed_mask_patch[id]
                        pos = moving_image_patch_position[id]
                        moving_boundary_patch1 = moving_boundary_patch[id]
                        fixed_boundary_patch1 = fixed_boundary_patch[id]
                        moving_image_batch_patch.append(moving_image_patch1)
                        fixed_image_batch_patch.append(fixed_image_patch1)
                        moving_mask_batch_patch.append(moving_mask_patch1)
                        fixed_mask_batch_patch.append(fixed_mask_patch1)
                        position.append(pos)
                        moving_boundary_batch_patch.append(moving_boundary_patch1)
                        fixed_boundary_batch_patch.append(fixed_boundary_patch1)
                    moving_image_batch_patch = np.array(moving_image_batch_patch)
                    fixed_image_batch_patch = np.array(fixed_image_batch_patch)
                    moving_mask_batch_patch = np.array(moving_mask_batch_patch)
                    fixed_mask_batch_patch = np.array(fixed_mask_batch_patch)
                    position = np.array(position)
                    moving_boundary_batch_patch = np.array(moving_boundary_batch_patch)
                    fixed_boundary_batch_patch = np.array(fixed_boundary_batch_patch)
                    moving_image_batch_patch = moving_image_batch_patch[..., np.newaxis]
                    fixed_image_batch_patch = fixed_image_batch_patch[..., np.newaxis]
                    yield ([moving_image_batch_patch, fixed_image_batch_patch,position,moving_mask_batch_patch,fixed_mask_batch_patch,moving_boundary_batch_patch,fixed_boundary_batch_patch],
                          [flow,fixed_image_batch_patch,fixed_image_batch_patch,fixed_image_batch_patch,fixed_image_batch_patch,y_label,y_label,y_true,y_true,y_true,y_true,y_true,y_true,y_true,y_true])
