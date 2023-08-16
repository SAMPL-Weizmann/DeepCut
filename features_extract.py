import numpy as np


def deep_features(image_tensor, extractor, layer, facet, bin: bool = False, include_cls: bool = False, device='cuda'):
    """
    Extract descriptors from pretrained DINO model; Create an adj matrix from descriptors
    @param image_tensor: Tensor of size (batch, height, width)
    @param extractor: Initialized model to extract descriptors from
    @param layer: Layer to extract the descriptors from
    @param facet: Facet to extract the descriptors from (key, value, query)
    @param bin: apply log binning to the descriptor. default is False.
    @param include_cls: To include CLS token in extracted descriptor
    @param device: Training device
    @return: W: adjacency matrix, F: feature matrix, D: row wise diagonal of W
    """

    # images to deep_features.
    # input is a tensor of size batch X height X width,
    # output is size: batch X 1 X height/patchsize * width/patchsize X deep_features size
    deep_features = extractor.extract_descriptors(image_tensor.to(device), layer, facet, bin, include_cls).cpu().numpy()

    # batch X height/patchsize * width/patchsize X deep_features size
    deep_features = np.squeeze(deep_features, axis=1)

    # batch * height/patchsize * width/patchsize X deep_features size
    deep_features = deep_features.reshape((deep_features.shape[0] * deep_features.shape[1], deep_features.shape[2]))

    # deep_features size X batch * (height* width/(patchsize ^2))
    return deep_features