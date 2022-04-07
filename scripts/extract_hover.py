import sys
# sys.path.append('../')

import numpy as np
import os
import glob
import scipy.io as sio
import pandas as pd
from imageio import imread
from histomicstk.features.compute_nuclei_features import compute_nuclei_features
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine

# get a random image
def extract_histomics_features(tile_mask_path, tile_path):
    image_mat_list = glob.glob(tile_mask_path + "*.mat")
    features_df_res = []
    num_tiles = len(image_mat_list)
    for i in range(num_tiles):
        # get the corresponding `.mat` file
        image_mat_file = image_mat_list[i]
        result_mat = sio.loadmat(image_mat_file)

        inst_map = result_mat['inst_map']
        inst_type = result_mat['inst_type']

        ### obtain the basename
        basename = os.path.basename(image_mat_file)
        image_ext = basename.split('.')[-1]
        basename = basename[:-(len(image_ext)+1)]
        inst_map_array = np.array(inst_map)

        ### run the color deconvolution on the png image
        tile_file = os.path.join(tile_path, basename)+".png"
        im = imread(tile_file)
        im_rgb = np.uint8(im[..., :3])
        stains, _, _ = color_deconvolution_routine(im_rgb=im_rgb)

        ### run the neuclei feature extraction
        features_df = compute_nuclei_features(im_label=inst_map_array,
                                                im_nuclei=stains[:, :, 0].copy(),
                                                im_cytoplasm=stains[:, :, 1].copy())
        if (features_df.shape[0] < 1):
            print(i, basename)
        else:
            features_df['predict_type'] = inst_type[:, 0]
            features_df['spot_id'] = np.repeat(basename, features_df.shape[0], axis=0)
            features_df_res.append(features_df)

    features_df_res = pd.concat(features_df_res)
    return features_df_res


if __name__ == "__main__":
    # PanNuke 151676
    tile_mask_path = "tests/LIBD/151676/hover_out_panuke/mat/"
    tile_path = "tests/LIBD/151676/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path=tile_mask_path, tile_path=tile_path)
    saved_features_df_file = "tests/LIBD/151676/hover_out_panuke/htk_PanNuke_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)

    # PanNuke
    tile_mask_path = "tests/HER2ST/B1/hover_out_pannuke/mat/"
    tile_path = "tests/HER2ST/B1/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path=tile_mask_path, tile_path=tile_path)
    saved_features_df_file = "tests/HER2ST/B1/hover_out_pannuke/htk_PanNuke_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)

    # MoNuSAC
    tile_mask_path = "tests/HER2ST/B1/hover_out_monusac/mat/"
    tile_path = "tests/HER2ST/B1/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path=tile_mask_path, tile_path=tile_path)
    saved_features_df_file = "tests/HER2ST/B1/hover_out_monusac/htk_MoNuSAC_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)


    # PanNuke
    tile_mask_path = "tests/HER2ST/A1/hover_out/mat/"
    tile_path = "tests/HER2ST/A1/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path = tile_mask_path, tile_path = tile_path)
    saved_features_df_file = "tests/HER2ST/A1/hover_out/htk_PanNuke_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)

    # MoNuSAC
    tile_mask_path = "tests/HER2ST/A1/hover_out_monusac/mat/"
    tile_path = "tests/HER2ST/A1/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path = tile_mask_path, tile_path = tile_path)
    saved_features_df_file = "tests/HER2ST/A1/htk_MoNuSAC_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)

    # CoNSeP
    tile_mask_path = "tests/HER2ST/A1/hover_out_consep/mat/"
    tile_path = "tests/HER2ST/A1/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path = tile_mask_path, tile_path = tile_path)
    saved_features_df_file = "tests/HER2ST/A1/htk_CoNSeP_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)


    tile_mask_path = "tests/HER2ST/A2/hover_out/mat/"
    tile_path = "tests/HER2ST/A2/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path = tile_mask_path, tile_path = tile_path)
    saved_features_df_file = "tests/HER2ST/A2/hover_out/htk_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)


    tile_mask_path = "tests/HER2ST/A3/hover_out/mat/"
    tile_path = "tests/HER2ST/A3/processed/sample_tiles/"
    features_df_res = extract_histomics_features(tile_mask_path = tile_mask_path, tile_path = tile_path)
    saved_features_df_file = "tests/HER2ST/A3/hover_out/htk_features.csv"
    features_df_res.to_csv(saved_features_df_file, index=False, header=True)


    # features_df_res.to_csv()
    # morphy_df = extract_histomics_features(tile_mat_path)
    # print(morphy_df)
    #
    #
    one_tile_mat_path = "tests/HER2ST/A2/hover_out/mat/11x26.mat"
    result_mat = sio.loadmat(one_tile_mat_path)
    inst_map = result_mat['inst_map']
    inst_type = result_mat['inst_type']
    inst_map_array = np.array(inst_map)
    inst_type_array = np.array(inst_type)
    import cv2

    image_file = "tests/HER2ST/A2/processed/sample_tiles/11x26.png"
    image = cv2.imread(image_file)
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay_file = "tests/HER2ST/A2/hover_out/overlay/11x26.png"
    overlay_img = cv2.imread(overlay_file)
    overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)


    import json
    json_file_path = "tests/HER2ST/A2/hover_out/json/11x26.json"
    bbox_list = []
    centroid_list = []
    contour_list = []
    type_list = []

    with open(json_file_path) as json_file:
        data = json.load(json_file)
        mag_info = data['mag']
        nuc_info = data['nuc']
        for inst in nuc_info:
            inst_info = nuc_info[inst]
            inst_centroid = inst_info['centroid']
            centroid_list.append(inst_centroid)
            inst_contour = inst_info['contour']
            contour_list.append(inst_contour)
            inst_bbox = inst_info['bbox']
            bbox_list.append(inst_bbox)
            inst_type = inst_info['type']
            type_list.append(inst_type)

    rand_nucleus = np.random.randint(0, len(centroid_list))
    rand_centroid = centroid_list[rand_nucleus]
    rand_bbox = bbox_list[rand_nucleus]
    rand_contour = contour_list[rand_nucleus]

    # draw the overlays
    overlay = image.copy()
    overlay = cv2.drawContours(overlay.astype('uint8'), [np.array(rand_contour)], -1, (255, 255, 0), 1)
    overlay = cv2.circle(overlay.astype('uint8'),
                         (np.round(rand_centroid[0]).astype('int'), np.round(rand_centroid[1]).astype('int')), 3,
                         (0, 255, 0), -1)
    overlay = cv2.rectangle(overlay.astype('uint8'), (rand_bbox[0][1], rand_bbox[0][0]),
                            (rand_bbox[1][1], rand_bbox[1][0]), (255, 0, 0), 1)

    import matplotlib.pyplot as plt
    pad = 30
    crop1 = rand_bbox[0][0] - pad
    if crop1 < 0:
        crop1 = 0
    crop2 = rand_bbox[1][0] + pad
    if crop2 > overlay.shape[0]:
        crop2 = overlay.shape[0]
    crop3 = rand_bbox[0][1] - pad
    if crop3 < 0:
        crop3 = 0
    crop4 = rand_bbox[1][1] + pad
    if crop4 > overlay.shape[1]:
        crop4 = overlay.shape[1]
    crop_overlay = overlay[crop1:crop2, crop3:crop4, :]
    plt.figure(figsize=(10, 10))

    plt.imshow(crop_overlay)
    plt.axis('off')
    plt.title('Overlay', fontsize=25)
    plt.show()


    import matplotlib.pyplot as plt

    plt.figure(figsize=(40, 20))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image', fontsize=40)

    plt.subplot(1, 3, 2)
    plt.imshow(inst_map_array)
    plt.axis('off')
    plt.title('Instance Map', fontsize=40)

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis('off')
    plt
    plt.title('Overlay', fontsize=40)

    plt.show()

