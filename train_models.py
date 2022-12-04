import argparse

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from GNNModels.Models import *
from GNNModels.Train import *



parser = argparse.ArgumentParser(description='Training GNN Models and Saving checkpoints')
parser.add_argument('--model', required=True,
                    help='provide model name from GCN,GAT,GNNGraphConv,GCN_3L')
parser.add_argument('--dataset',required=True,
                    help='provide dataset name from PROTEINS, MUTAG, Cora, CiteSeer')
parser.add_argument('--checkpoints',required=False,default='',
                    help='location to store checkpoints')

args = parser.parse_args()

# obtain the arguments to the python file
dataset_name=args.dataset
model_name=args.model
checkpoint_path=args.checkpoints

dataset=None
data=None

print("NOTE : This will overwrite checkpoints in new_checkpoints/")

print(dataset_name)
print(model_name)
print(checkpoint_path)

if dataset_name in ['Cora','CiteSeer']:
    dataset = Planetoid(root='GNNModels/data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]  # Get the first graph object.
elif dataset_name in ['MUTAG','PROTEINS']:
    dataset = TUDataset(root='GNNModels/data/TUDataset', name='PROTEINS')
else:
    print("Not a valid dataset name")

if model_name=='GCN':
    model = GCN(hidden_channels=16,num_features=dataset.num_features,num_classes=dataset.num_classes)
    params={'lr':0.01,'weight_decay':5e-4,'epochs':300,'verbose':True,'save_wts':checkpoint_path}
    #new_checkpoints/{}_{}_epochs=300.pt
    TrainModel(model,data,params,'NC')
elif model_name=='GAT':
    model= GAT(hidden_channels=8, num_features=dataset.num_features,num_classes=dataset.num_classes,heads=8)
    params={'lr':0.005,'weight_decay':5e-4,'epochs':300,'verbose':True,'save_wts':checkpoint_path}
    # new_checkpoints/GAT_PubMed_epochs=300.pt
    TrainModel(model,data,params,'NC')
elif model_name=='GCN_3L':
    model = GCN_3L(hidden_channels=64,num_features=dataset.num_node_features,num_classes=dataset.num_classes)
    params={'lr':0.01,'epochs':150,'verbose':True,'save_wts':checkpoint_path}
    #new_checkpoints/GCN_3L_PROTEINS_epochs=300.pt
    TrainModel(model,dataset,params,type='GC')
elif model_name=='GNNGraphConv':
    model = GNNGraphConv(hidden_channels=64,num_features=dataset.num_node_features,num_classes=dataset.num_classes)
    params={'lr':0.01,'epochs':300,'verbose':True,'save_wts':checkpoint_path}
    # new_checkpoints/GNNGraphConv_PROTEINS_epochs=300.pt
    TrainModel(model,dataset,params,type='GC')
else :
    print("Not a valid model name")
