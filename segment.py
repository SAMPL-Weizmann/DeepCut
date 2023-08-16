from bilateral_solver import bilateral_solver_output
from features_extract import deep_features
from torch_geometric.data import Data
from extractor import ViTExtractor
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import util
import os


def GNN_seg(mode, cut, alpha, epoch, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer,
            stride, device):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param cut: chosen clustering functional: NCut==1, CC==0
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
    @param device: Device to use ('cuda'/'cpu')
    """
    ##########################################################################################
    # Dino model init
    ##########################################################################################
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    # VIT small feature dimension, with or without log bin
    if not log_bin:
        feats_dim = 384
    else:
        feats_dim = 6528

    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    ##########################################################################################
    # GNN model init
    ##########################################################################################
    # import cutting gnn model if cut == 0 NCut else CC
    if cut == 0: 
        from gnn_pool import GNNpool 
    else: 
        from gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        torch.save(model2.state_dict(), 'model2.pt')
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        torch.save(model3.state_dict(), 'model3.pt')
        model3.train()

    ##########################################################################################
    # Iterate over files in input directory and apply GNN segmentation
    ##########################################################################################
    for filename in tqdm(os.listdir(in_dir)):
        # If not image, skip
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue
        # if file already processed
        if os.path.exists(os.path.join(out_dir, filename.split('.')[0] + '.txt')):
            continue
        if os.path.exists(os.path.join(out_dir, filename)):
            continue

        ##########################################################################################
        # Data loading
        ##########################################################################################
        # loading images
        image_tensor, image = util.load_data_img(os.path.join(in_dir, filename), res)
        # Extract deep features, from the transformer and create an adj matrix
        F = deep_features(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
        W = util.create_adj(F, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model.parameters(), lr=0.001)

        ##########################################################################################
        # GNN pass
        ##########################################################################################
        for _ in range(epoch[0]):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()

        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)

        ##########################################################################################
        # Post-processing Connected Component/bilateral solver
        ##########################################################################################
        mask0, S = util.graph_to_mask(S, cc, stride, image_tensor, image)
        # apply bilateral solver
        if bs:
            mask0 = bilateral_solver_output(image, mask0)[1]

        if mode == 0:
            util.save_or_show([image, mask0, util.apply_seg_map(image, mask0, 0.7)], filename, out_dir ,save)
            continue

        ##########################################################################################
        # Second pass on foreground
        ##########################################################################################
        # extracting foreground sub-graph
        sec_index = np.nonzero(S).squeeze(1)
        F_2 = F[sec_index]
        W = util.create_adj(F_2, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W_2, F_2)
        data_2 = Data(node_feats, edge_index, edge_weight).to(device)
        # re-init weights and optimizer for every image
        model2.load_state_dict(torch.load('./model2.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model2.parameters(), lr=0.001)

        ####################################################
        # GNN pass
        ####################################################
        for _ in range(epoch[1]):
            opt.zero_grad()
            A_2, S_2 = model2(data_2, torch.from_numpy(W_2).to(device))
            loss = model2.loss(A_2, S_2)
            loss.backward()
            opt.step()

        # fusing subgraph and original graph
        S_2 = S_2.detach().cpu()
        S_2 = torch.argmax(S_2, dim=-1)
        S[sec_index] = S_2 + 3

        mask1, S = util.graph_to_mask(S, cc, stride, image_tensor, image)

        if mode == 1:
            util.save_or_show([image, mask1, util.apply_seg_map(image, mask1, 0.7)],filename,out_dir, save)
            continue

        ##########################################################################################
        # Second pass background
        ##########################################################################################
        # extracting background sub-graph
        sec_index = np.nonzero(S == 0).squeeze(1)
        F_3 = F[sec_index]
        W = util.create_adj(F_3, cut, alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W_3, F_3)
        data_3 = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model3.load_state_dict(torch.load('./model3.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model3.parameters(), lr=0.001)
        for _ in range(epoch[2]):
            opt.zero_grad()
            A_3, S_3 = model3(data_3, torch.from_numpy(W_3).to(device))
            loss = model3.loss(A_3, S_3)
            loss.backward()
            opt.step()

        # fusing subgraph and original graph
        S_3 = S_3.detach().cpu()
        S_3 = torch.argmax(S_3, dim=-1)
        S[sec_index] = S_3 + foreground_k + 5

        mask2, S = util.graph_to_mask(S, cc, stride, image_tensor, image)
        if bs:
            mask_foreground = mask0
            mask_background = np.where(mask2 != foreground_k + 5, 0, 1)
            bs_foreground = bilateral_solver_output(image, mask_foreground)[1]
            bs_background = bilateral_solver_output(image, mask_background)[1]
            mask2 = bs_foreground + (bs_background * 2)

        util.save_or_show([image, mask2, util.apply_seg_map(image, mask2, 0.7)], filename, out_dir, save)


if __name__ == '__main__':
    ################################################################################
    # Mode
    ################################################################################
    # mode == 0 Single stage segmentation
    # mode == 1 Two stage segmentation for foreground
    # mode == 2 Two stage segmentation on background and foreground
    mode = 0
    ################################################################################
    # Clustering function
    ################################################################################
    # NCut == 0
    # CC == 1
    # alpha = k-sensetivity paremeter
    cut = 0
    alpha = 3
    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per stage [mode0,mode1,mode2]
    epochs = [10, 100, 10]
    # Number of steps per image
    step = 1
    # Number of clusters
    K = 2
    ################################################################################
    # Processing parameters
    ################################################################################
    # Show only largest component in segmentation map (for k == 2)
    cc = False
    # apply bilateral solver
    bs = False
    # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
    log_bin = False
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    # Resolution for dino input, higher res != better performance as Dino was trained on (224,224) size images
    res = (280, 280)
    # stride for descriptor extraction
    stride = 8
    # facet fo descriptor extraction (key/query/value)
    facet = 'key'
    # layer to extract descriptors from
    layer = 11
    ################################################################################
    # Data parameters
    ################################################################################
    # Directory of image to segment
    in_dir = './images/single/'
    out_dir = './results/'
    save = False
    ################################################################################
    # Check for mistakes in given arguments
    assert not(K != 2 and cc), 'largest connected component only available for k == 2'

    # if CC set maximum number of clusters
    if cut == 1:
        K = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(mode, cut, alpha, epochs, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer, stride,
            device)