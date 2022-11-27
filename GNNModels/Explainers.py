from GNNModels.Models import *

import torch
from tqdm import tqdm
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from graphxai.explainers import GNNExplainer, PGExplainer, IntegratedGradExplainer, PGMExplainer
# the ones below we want to use from different libraries
from graphxai.explainers import GNN_LRP, CAM
import pickle
import datetime 


# GNN Explainer - discrete mask of node imp, soft mask of edge imp
def gnn_imp_nodes(gnnexp,data,node_idx):

    node_exp = gnnexp.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index)

    imp_nodes = []

    for k in node_exp.node_reference.keys():

        if node_exp.node_imp[node_exp.node_reference[k]].item() == 1:

            imp_nodes.append(k)

    return imp_nodes

# PGE Explainer - discrete maks of node imp, discrete mask of edge imp

def pge_imp_nodes(pgex,data,node_idx):
    if pgex==None: 
        return
    node_exp = pgex.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index)

    imp_nodes = []

    for k in node_exp.node_reference.keys():

        if node_exp.node_imp[node_exp.node_reference[k]].item() == 1:

            imp_nodes.append(k)

    return imp_nodes

# Integrated gradients - soft mask of edge imp
def ig_imp_nodes(igex,data,node_idx):

    node_exp = igex.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index, y = data.y)

    imp_nodes = []

    mask = torch.sigmoid(node_exp.node_imp) >= 0.5

    for k in node_exp.node_reference.keys():

        if mask[node_exp.node_reference[k]].item() == 1:
        
            imp_nodes.append(k)

    return imp_nodes

# PGME Explainer - discrete mask of node imp, randomised, can get ranking as well by asking for top 1 then 2 and so on
def pgm_imp_nodes(pgm,data,node_idx, top = None):

    np.random.seed(1998)

    if top == None:

        node_exp = pgm.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index)

    else: 

        node_exp = pgm.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index, top_k_nodes=top)

    imp_nodes = []

    for k in node_exp.node_reference.keys():

        if node_exp.node_imp[node_exp.node_reference[k]].item() == 1:
        
            imp_nodes.append(k)

    return imp_nodes

# CAM - soft mask of node importanct

def cam_imp_nodes(camex,data,node_idx):
    if camex==None:
        return

    node_exp = camex.get_explanation_node(node_idx = node_idx, x = data.x, edge_index = data.edge_index, y = data.y)

    imp_nodes = []

    mask = torch.sigmoid(node_exp.node_imp) >= 0.5

    for k in node_exp.node_reference.keys():

        if mask[node_exp.node_reference[k]].item() == 1:
        
            imp_nodes.append(k)

    return imp_nodes



def createExplanations(model_name,dataset_name,type):
    '''
    model = [ GCN, GAT, GNNConv, GCN_3L]
    dataset= [ 'Cora','PubMed','CiteSeer','MUTAG','PROTIENS']
    type=['GC','NC'] 

    '''
    assert(type=='NC') # currently works only for NC   
    assert(model_name in ['GCN','GAT'])
    assert(dataset_name in ['Cora','PubMed','CiteSeer'])

    # variables depending on dataset and model required to run explainers
    criterion=None
    emb_layer_name=''
    dataset=None
    data=None
    model=None
    in_channels=2

    if type=='NC':
        dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        criterion=torch.nn.CrossEntropyLoss()

        if model_name=='GCN':
            model=get_model_pretrained('GCN',dataset_name)
            emb_layer_name='conv2'
        elif model_name=='GAT':
            model=get_model_pretrained('GAT',dataset_name)
            # emb_layer_name='conv2' # GAT is not yet accomadated for in PGExpl you have to give the input channle manualy
            emb_layer = list(model.modules())[-3]
            in_channels=3* emb_layer.heads

    igex = IntegratedGradExplainer(model, criterion=criterion)
    pgm = PGMExplainer(model, explain_graph=False)
    camex = None


    pgex=None
    if emb_layer_name!='':
    # needs name of emb layer of the model
        pgex = PGExplainer(model, emb_layer_name =emb_layer_name ,  max_epochs = 500, lr = 0.01)
        pgex.train_explanation_model(data)
        camex=CAM(model)
    else :
        # pgex = PGExplainer(model, in_channels=in_channels ,  max_epochs = 500, lr = 0.01)
        pgex=None
        camex=None

    gnnexp = GNNExplainer(model)


    out = model(data.x, data.edge_index)

    imp_nodes_ig = {}
    imp_nodes_gnn = {}
    imp_nodes_pge = {}
    imp_nodes_pgm = {}
    imp_nodes_cam = {}
    node_indices=[]
    for node_idx in tqdm((data.test_mask == True).nonzero()): 

        if out[node_idx].argmax() != data.y[node_idx]: # only accross all the correctly classified nodes
            continue
        node_indices.append(node_idx)
        imp_nodes_ig[node_idx] = ig_imp_nodes(igex,data,node_idx)
        imp_nodes_gnn[node_idx] = gnn_imp_nodes(gnnexp,data,node_idx)
        imp_nodes_pge[node_idx] = pge_imp_nodes(pgex,data,node_idx)
        imp_nodes_pgm[node_idx] = pgm_imp_nodes(pgm,data,node_idx.item())
        imp_nodes_cam[node_idx] = cam_imp_nodes(camex,data,node_idx)



    # define dictionary
    expl_to_save = {'node_indices':node_indices,
            'ig' : imp_nodes_ig, 
            'gnn' : imp_nodes_gnn, 
            'pge' : imp_nodes_pge,
            'pgm':imp_nodes_pgm,
            'cam':imp_nodes_cam}

    # create a binary pickle file 
    
    now = datetime.datetime.now()
    timestamp_str=now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))
    
    with open("Saved_Explanations/Explanations_{}_{}_{}.pkl".format(model_name,dataset_name,timestamp_str),"wb") as f:
        # write the python object (dict) to pickle file
        pickle.dump(expl_to_save,f)
