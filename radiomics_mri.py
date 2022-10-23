import nibabel as nib
import radiomics
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D, featureextractor
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import glob
import re
import os
import matplotlib.animation as animation
import json
import six
import SimpleITK as sitk

"""
GD-enhancing tumor (ET — label 4)
peritumoral edema (ED — label 2)
necrotic and non-enhancing tumor core (NCR/NET — label 1)
classes = ['gd_enhancing', 'edema', 'necrotic_area']

mask pixel value = 1
"""

json_open = open('Script/empty.json', 'r')
results_json = json.load(json_open)

files = os.listdir('D:/UCSF-PDGM-nifti')
files_dir = [f for f in files if os.path.isdir(
    os.path.join('D:/UCSF-PDGM-nifti/', f))]

for dir_name in files_dir:

    params = {}
    params['bandwidth'] = 20
    params['sigma'] = [1, 2, 3]
    params['verbose'] = True
    params['label'] = 1

    params2d = {}
    params2d['bandwidth'] = 20
    params2d['sigma'] = [1, 2, 3]
    params2d['verbose'] = True
    params2d['label'] = 1
    params2d['force2D'] = True
    params2d['correctMask'] = True

    image_dir = dir_name
    patient_id = re.findall('(.*)_', image_dir)[0]
    print(patient_id)
    modality = 'T1.nii.gz'
    seg = 'tumor_segmentation.nii.gz'

    img = nib.load('D:/UCSF-PDGM-nifti/' + image_dir +
                   '/' + patient_id + '_' + modality)
    msk = nib.load('D:/UCSF-PDGM-nifti/' + image_dir +
                   '/' + patient_id + '_' + seg)
    img_data = img.get_fdata()
    msk_data = msk.get_fdata()
    msk_data = np.where(msk_data > 0, 1.0, 0)
    msk_data = msk_data.astype(np.uint32)

    l, m, n = img_data.shape

    for i in range(n):
        if np.count_nonzero(msk_data[:, :, i] > 0) > 16:
            n_overzero = np.count_nonzero(msk_data[:, :, i] > 0)
            image = sitk.GetImageFromArray(img_data[:, :, i])
            mask = sitk.GetImageFromArray(msk_data[:, :, i])

            bb, correctedMask = imageoperations.checkMask(image, mask)
            if correctedMask is not None:
                mask = correctedMask

            results_json['ID'].append(patient_id)
            results_json['Filename'].append(patient_id + '_{}'.format(i))
            results_json['pixels'].append(n_overzero)

            """ First Order Features """
            features = radiomics.firstorder.RadiomicsFirstOrder(
                image, mask, **params)
            features.enableAllFeatures()
            features.enabledFeatures['StandardDeviation'] = True
            results = features.execute()
            for (key, val) in six.iteritems(results):
                print(key)
                results_json['FirstOrder'][key].append(val.tolist())

            """ Shape Features (2D) """
            #image_shape2d, mask_shape2d = shape2d_im_constructor(path, mask_path)
            features = radiomics.shape2D.RadiomicsShape2D(
                image, mask, **params2d)
            features.enableAllFeatures()
            features.enabledFeatures['SphericalDisproportion'] = True
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['Shape2D'][key].append(val.tolist())

            """ Gray Level Co-occurrence Matrix (GLCM) Features """
            features = radiomics.glcm.RadiomicsGLCM(image, mask, **params)
            features.enableAllFeatures()
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['GLCM'][key].append(val.tolist())

            """ Gray Level Run Length Matrix (GLRLM) Features """
            features = radiomics.glrlm.RadiomicsGLRLM(image, mask, **params)
            features.enableAllFeatures()
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['GLRLM'][key].append(val.tolist())

            """ Gray Level Size Zone Matrix (GLSZM) Features """
            features = radiomics.glszm.RadiomicsGLSZM(image, mask, **params)
            features.enableAllFeatures()
            features.enabledFeatures['LargeAreaHighGrayLevelEmphasis'] = True
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['GLSZM'][key].append(val.tolist())

            """ Neighbouring Gray Tone Difference Matrix (NGTDM) Features """
            features = radiomics.ngtdm.RadiomicsNGTDM(image, mask, **params)
            features.enableAllFeatures()
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['NGTDM'][key].append(val.tolist())

            """ Gray Level Dependence Matrix (GLDM) Features """
            features = radiomics.gldm.RadiomicsGLDM(image, mask, **params)
            features.enableAllFeatures()
            results = features.execute()
            for (key, val) in six.iteritems(results):
                results_json['GLDM'][key].append(val.tolist())

    with open("./T1_results.json", 'w') as outfile:
        json.dump(results_json, outfile, indent=4)


"""
for i in range(n):
    r = re.findall(r'\_(.*?)\.', files[i])[0]
    print(r)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title(r)

    imgs = []
    img = nib.load(image_dir + '/' + files[i])
    img_data = img.get_fdata()
    print(img_data.shape)

    # plt.imshow(sitk.GetArrayFromImage(image))
    # plt.show()

    # plt.imshow(sitk.GetArrayFromImage(mask))
    # plt.show()

    #img_itk = sitk.ReadImage(image_dir + '/' + patient_id + '_' + modality)
    #msk_itk = sitk.ReadImage(image_dir + '/' + patient_id + '_' + seg, sitk.sitkUInt32)
"""
