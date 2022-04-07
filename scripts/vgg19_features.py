from torchvision import models
import torch
from tqdm import tqdm
import os
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from histolab.slide import Slide
from histolab.types import CoordinatePair



def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """

    ## get the batch_size, depth, height, and width of the Tensor
    ## reshape it, so we're multiplying the features for each channel
    ## calculate the gram matrix

    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(d, -1)

    gram = torch.mm(tensor, tensor.t())

    return gram


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1',
                  '34': 'conv5_4',
                  '36': 'Maxpool'}

    ## -- do not need to change the code below this line -- ##
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features



def extract_vgg19(slice, img_path, tile_size=64, layer_name = "conv5_4"):

    """
    Extract VGG19 features from tiled tile images.
    """

    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg.to(device)

    if os.path.exists(img_path):
        processed_path = img_path
    else:
        processed_path = os.getcwd()
    slice_Img = Slide(img_path, processed_path=processed_path)
    positions = slice.obsm['pixel']
    num_Tiles = positions.shape[0]

    feature_res = []
    spot_ids = []
    feature_res = []
    spot_ids = []

    for i in tqdm(range(num_Tiles)):
        tmp_coords = positions[i]
        tmp_coords = list(map(float, tmp_coords))
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
        # vgg19 features
        try:
            size = max(pil_img.size)
            in_transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            pil_img = in_transform(pil_img)[:3, :, :].unsqueeze(0)
            pil_img.to(device)
            vgg_features = get_features(pil_img, vgg)
            vgg_grams = {layer: gram_matrix(vgg_features[layer]) for layer in vgg_features}
            vgg_mat = vgg_grams[layer_name]
            pca_u, _, _ = torch.pca_lowrank(vgg_mat)
            pca_u_features = pca_u.flatten()
            pca_u_features = pca_u_features.detach().numpy()
            feature_res.append(pca_u_features)
            # spot ids
            spot = slice.obs.index[i]
            spot_ids.append(spot)
        except ZeroDivisionError:
            pass
    slice.obsm['VGG'] = np.array(feature_res)
    return feature_res, spot_ids


#

# import numpy as np
# import os
# from tqdm import tqdm
#
#
def extract_vgg19_keras(path_to_tiles):
    from keras.preprocessing import image
    from keras.models import Model
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input
    """
    Extract VGG19 features from tiled tile images.
    """
    model = VGG19(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('flatten').output)
    layer_names = [layer.name for layer in model.layers]


    feature_res = []
    spot_ids = []
    file_names = os.listdir(path_to_tiles)
    for file_name in tqdm(file_names):
        try:
            # vgg16 features
            one_tile_path = os.path.join(path_to_tiles, file_name)
            tile = image.load_img(one_tile_path, target_size=(224, 224))
            tile = image.img_to_array(tile)
            tile = np.expand_dims(tile, axis=0)
            process_tile = preprocess_input(tile)
            features = model.predict(process_tile, batch_size=1)
            features = features.flatten()
            feature_res.append(features)
            # spot ids
            spot_name = file_name.split(".")[0]
            spot_ids.append(spot_name)
        except ZeroDivisionError:
            pass

    return feature_res, spot_ids


