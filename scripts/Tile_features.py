from tensorflow.keras.applications.densenet import DenseNet121
import numpy as np
import os
import sys

import torch
from PIL import Image
from histolab.slide import Slide
from histolab.types import CoordinatePair
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from pytorch_pretrained_vit import ViT
from torchvision import transforms
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def Tile_Features(slice, img_path, tile_size=64, method="ViT", saved_path=None):
    if os.path.exists(img_path):
        processed_path = img_path
    else:
        processed_path = os.getcwd()
    slice_Img = Slide(img_path, processed_path=processed_path)
    positions = slice.obsm['pixel']
    num_Tiles = positions.shape[0]

    if method == "ViT":
        model = ViT('B_16_imagenet1k', pretrained=True)
        del model.fc
    elif method == "ResNet":
        model = ResNet50(weights='imagenet', include_top=True)
        model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)
    elif method == "DenseNet":
        model = DenseNet121(weights="imagenet", include_top=True)
        model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)
    else:
        sys.exit('use the correct model')
    # remove the last layer
    feature_res = []
    spot_ids = []

    for i in tqdm(range(num_Tiles)):
        tmp_coords = positions[i]
        tmp_coords = list(map(int, tmp_coords))
        ix1 = tmp_coords[0] - tile_size
        iy1 = tmp_coords[1] - tile_size
        ix2 = tmp_coords[0] + tile_size
        iy2 = tmp_coords[1] + tile_size
        tmp_coords_vec = [ix1, iy1, ix2, iy2]
        tile = slice_Img.extract_tile(
            coords=CoordinatePair(*tmp_coords_vec),
            level=0,
            tile_size=(2 * tile_size, 2 * tile_size)
        )
        pil_img = tile.image.convert("RGB")
        if method == "ViT":
            pil_img = pil_img.resize((384, 384))  # default is te BICUBIC upstreaming
            img = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])(pil_img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img).squeeze(0)
                features = torch.squeeze(outputs)
                features = features.mean(0)
                features = features.detach().numpy()
                feature_res.append(features)

                # spot ids
                spot = slice.obs.index[i]
                spot_ids.append(spot)

        elif method=="ResNet":
            tile = pil_img.resize((224, 224)) # default is te BICUBIC upstreaming
            tile = np.expand_dims(tile, axis=0)
            process_tile = preprocess_input(tile)
            features = model.predict(process_tile, batch_size=1)
            features = features.flatten()
            feature_res.append(features)

            spot = slice.obs.index[i]
            spot_ids.append(spot)

        else:
            tile = pil_img.resize((224, 224))  # default is te BICUBIC upstreaming
            tile = np.expand_dims(tile, axis=0)
            process_tile = preprocess_input(tile)
            features = model.predict(process_tile, batch_size=1)
            features = features.flatten()
            feature_res.append(features)
            spot = slice.obs.index[i]
            spot_ids.append(spot)


    slice.obsm[method] = np.array(feature_res)
    print("done")
    # save the results
    if saved_path is None:
        return feature_res, spot_ids
    else:
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        saved_file = os.path.join(saved_path, method) + ".npz"
        np.savez_compressed(saved_file,
                            feature_res=feature_res, spot_res=spot_ids)



