from nibabel import load
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import glob
import re
import os

path_pdir = 'C:/Users/kuma/Dropbox/brain_mri/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations'

save_dir = 'C:/Users/kuma/Dropbox/brain_mri/images/'

path_dir = []
path_ann = []

for f in os.listdir(path_pdir):
  if os.path.isdir(os.path.join(path_pdir, f)):
    path_dir.append(f)


for i in path_dir:
    path_niz = glob.glob(path_pdir + '/' + i + '/*.nii.gz')
    if len(path_niz) == 6:
        for j in path_niz:
            if 'GlistrBoost_Manually' in j:
                str_app = re.match(r'(.*?)GlistrBoost.*', j).group(1)
                path_ann.append(str_app)
    elif len(path_niz) == 5:
        for j in path_niz:
            if 'GlistrBoost' in j:
                str_app = re.match(r'(.*?)GlistrBoost.*', j).group(1)
                path_ann.append(str_app)


for j in range(len(path_ann)):
    if os.path.exists(path_ann[j] + 'GlistrBoost_ManuallyCorrected.nii.gz'):
        path_annotation = path_ann[j] + 'GlistrBoost_ManuallyCorrected.nii.gz'
    else:
        path_annotation = path_ann[j] + 'GlistrBoost.nii.gz'

    path_flair = path_ann[j] + 'flair.nii.gz'
    path_t1 = path_ann[j] + 't1.nii.gz'
    path_t1Gd = path_ann[j] + 't1Gd.nii.gz'
    path_t2 = path_ann[j] + 't2.nii.gz'

    annotation = load(path_annotation)
    flair = load(path_flair)
    t1 = load(path_t1)
    t1Gd = load(path_t1Gd)
    t2 = load(path_t2)

    l, m, n = annotation.get_fdata().shape
    for i in range(n):
        ann_im = annotation.get_fdata()[8:232, 8:232, i]

        flair_im = flair.get_fdata()[8:232, 8:232, i]
        flair_im = (flair_im/np.max(flair_im))*255
        #flair_im_p = Image.fromarray(flair_im)
        #flair_im_p = flair_im_p.convert("RGB")
        #flair_im = np.array(flair_im_p)
        #plt.imshow(flair_im)
        #plt.show()

        t1_im = t1.get_fdata()[8:232, 8:232, i]
        t1_im = (t1_im/np.max(t1_im))*255
        #t1_im_p = Image.fromarray(t1_im)
        #t1_im_p = t1_im_p.convert("RGB")
        #t1_im = np.array(t1_im_p)
        #plt.imshow(t1_im)
        #plt.show()

        t1Gd_im = t1Gd.get_fdata()[8:232, 8:232, i]
        t1Gd_im = (t1Gd_im/np.max(t1Gd_im))*255
        #t1Gd_im_p = Image.fromarray(t1Gd_im)
        #t1Gd_im_p = t1Gd_im_p.convert("RGB")
        #t1Gd_im = np.array(t1Gd_im_p)
        #plt.imshow(t1Gd_im)
        #plt.show()

        t2_im = t2.get_fdata()[8:232, 8:232, i]
        t2_im = (t2_im/np.max(t2_im))*255
        #t2_im_p = Image.fromarray(t2_im)
        #t2_im_p = t2_im_p.convert("RGB")
        #t2_im = np.array(t2_im_p)
        #plt.imshow(t2_im)
        #plt.show()

        m_ann, n_ann  = ann_im.shape
        pixel_list = [ann_im[i, j] for i in range(m_ann) for j in range(n_ann)]
        if len(set(pixel_list)) > 3:
            np.save(save_dir + 'annotation/' + os.path.basename(path_ann[j]).replace('.niz.gz', '').replace('.', '_') + '{}'.format(i), ann_im)
            np.save(save_dir + 'flair/' + os.path.basename(path_ann[j]).replace('.niz.gz', '').replace('.', '_') + '{}'.format(i), flair_im)
            np.save(save_dir + 't1/' + os.path.basename(path_ann[j]).replace('.niz.gz', '').replace('.', '_') + '{}'.format(i), t1_im)
            np.save(save_dir + 't1Gd/' + os.path.basename(path_ann[j]).replace('.niz.gz', '').replace('.', '_') + '{}'.format(i), t1Gd_im)
            np.save(save_dir + 't2/' + os.path.basename(path_ann[j]).replace('.niz.gz', '').replace('.', '_') + '{}'.format(i), t2_im)
    print('{}/{}'.format(j+1, len(path_ann)), ann_im.shape, flair_im.shape, t1_im.shape, t1Gd_im.shape, t2_im.shape)

print('++++++++++ END ++++++++++')

"""
flair = 'C:/Users/takumau/Dropbox/brain_mri/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations\TCGA-02-0006/TCGA-02-0006_1996.08.23_GlistrBoost.nii.gz'
glb = 'C:/Users/takumau/Dropbox/brain_mri/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/TCGA-02-0006_1996.08.23_flair.nii.gz'


flair = 'C:/Users/kuma/Desktop/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/TCGA-02-0006_1996.08.23_GlistrBoost.nii.gz'
glb = 'C:/Users/kuma/Desktop/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/TCGA-02-0006_1996.08.23_flair.nii.gz'


flair = load(flair)
glb = load(glb)

print(flair.get_fdata().shape)
print(glb.get_fdata().shape)


for i in range(50,90):
    flair_im = flair.get_fdata()[:,:,i]
    flair_im = np.rot90(flair_im, 3)
    m, n = flair_im.shape
    pixel_list = [flair_im[i,j] for i in range(m) for j in range(n)]
    print(set(pixel_list))
    glb_im = glb.get_fdata()[:,:,i]
    glb_im = np.rot90(glb_im, 3)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(flair_im, cmap = "gray")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(glb_im, cmap = "gray")
    plt.axis('off')
    #img_np = img_np.rot90(img_np, 1)
    #flair_pil = Image.fromarray(flair)
    #print(image.get_fdata())
    plt.show()
"""
