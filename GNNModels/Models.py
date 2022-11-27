import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv,global_mean_pool,GraphConv


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(num_features, hidden_channels,heads)  # TODO
        self.conv2 = GATConv(hidden_channels*heads,num_classes,1)  # TODO

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels,num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN_3L(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super(GCN_3L, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels,num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GNNGraphConv(torch.nn.Module):
    def __init__(self, hidden_channels,num_features,num_classes):
        super(GNNGraphConv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_features,hidden_channels)  
        self.conv2 = GraphConv(hidden_channels,hidden_channels)  
        self.conv3 = GraphConv(hidden_channels,hidden_channels)  
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def get_trained_model_menu():
    pass
    # implement a print for a menu which shows avialble weights of trained models and corresponding datasets and type (graph or node classif)
    return

def get_model_pretrained(model_name,dataset_name,path=''):
    '''
        model_name = 'GCN', 'GAT' for graph classification
        dataset_name = Cora
    '''
    Datasets_specs={'Cora':{'num_features':1433,'num_classes':7},
                    'PubMed':{'num_features':500,'num_classes':3},
                    'CiteSeer':{'num_features':3703,'num_classes':6},
                    'MUTAG':{'num_features':7,'num_classes':2},
                    'PROTEINS':{'num_features':3,'num_classes':2}}
    # path='checkpoints/{}_{}'
    # model = ()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    model=None
    datasets_list=list(Datasets_specs.keys())
    if dataset_name not in  datasets_list:
        print("No weights available for this dataset")
        print("Available Datasets : ",datasets_list)
        return None

    if model_name not in ['GAT','GCN','GCN_3L','GNNGraphConv']:
        print("No weights available for this model")
        print("Available Datasets : ",['GAT','GCN','GCN_3L','GNNGraphConv'])
        return None

    if path=='':
        path='checkpoints/'

    if(model_name=='GAT'):
        model=GAT(hidden_channels=8, num_features=Datasets_specs[dataset_name]['num_features'],num_classes=Datasets_specs[dataset_name]['num_classes'],heads=8)
        model.load_state_dict(torch.load('{}GAT_{}_epochs=300.pt'.format(path,dataset_name)))
        model.eval()
    elif(model_name=='GCN'):
        model=GCN(hidden_channels=16, num_features=Datasets_specs[dataset_name]['num_features'],num_classes=Datasets_specs[dataset_name]['num_classes'])
        model.load_state_dict(torch.load('{}GCN_{}_epochs=300.pt'.format(path,dataset_name)))
        model.eval()
    elif(model_name=='GCN_3L'):
        model=GCN_3L(hidden_channels=64, num_features=Datasets_specs[dataset_name]['num_features'],num_classes=Datasets_specs[dataset_name]['num_classes'])
        model.load_state_dict(torch.load('{}GCN_3L_{}_epochs=300.pt'.format(path,dataset_name)))
        model.eval()
    elif(model_name=='GNNGraphConv'):
        model=GNNGraphConv(hidden_channels=64, num_features=Datasets_specs[dataset_name]['num_features'],num_classes=Datasets_specs[dataset_name]['num_classes'])
        model.load_state_dict(torch.load('{}GNNGraphConv_{}_epochs=300.pt'.format(path,dataset_name)))
        model.eval()
    
    return model