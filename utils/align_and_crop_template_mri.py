import nibabel as nib
import numpy as np

def center_align(file_src, file_dst, output_path):
    # Load source and destination images
    nii_dst = nib.load(file_dst)
    nii_src = nib.load(file_src)

    aff_dst = nii_dst.affine
    center_dst = aff_dst.dot(np.hstack((np.array(nii_dst.shape) / 2.0, [1])))

    aff_src = nii_src.affine
    rel_center_src = aff_src.dot(np.hstack((np.array(nii_src.shape) / 2.0, [0])))

    # Align the center
    center_diff = center_dst - rel_center_src
    aligned_aff_src = np.hstack((aff_src[:, :3], center_diff.reshape((4, 1))))
    nii_src.set_sform(aligned_aff_src, 2)

    # Get source and destination shapes
    shape_src = np.array(nii_src.shape)
    shape_dst = np.array(nii_dst.shape)

    # Compute padding and cropping
    pad_sizes = np.maximum(0, shape_dst - shape_src)  # Padding needed
    crop_sizes = np.maximum(0, shape_src - shape_dst)  # Cropping needed

    data_src = nii_src.get_fdata()

    # Apply padding if needed
    if np.any(pad_sizes > 0):
        pad_width = [(pad_sizes[i] // 2, pad_sizes[i] - pad_sizes[i] // 2) for i in range(3)]
        data_src = np.pad(data_src, pad_width, mode='constant', constant_values=0)  # Zero padding

    # Apply cropping if needed
    if np.any(crop_sizes > 0):
        crop_slices = [slice(crop_sizes[i] // 2, shape_src[i] - (crop_sizes[i] - crop_sizes[i] // 2)) for i in range(3)]
        data_src = data_src[crop_slices[0], crop_slices[1], crop_slices[2]]

    # Save the new image with aligned center and matched dimensions
    new_nii_src = nib.Nifti1Image(data_src, aligned_aff_src)
    nib.save(new_nii_src, output_path)

    print(f"Saved aligned and resized image to: {output_path}")


file_FDG = 'template/MNI_MRI.nii'
file_template = 'template/ROI_MNI_V7.nii'
file_conformed_FDG = 'template/MNI_MRI_conformed.nii'

'''
Normalize the FDG image to the template
'''

center_align(file_FDG, file_template, file_conformed_FDG)
