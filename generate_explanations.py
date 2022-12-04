from GNNExplainers.Explainers import *

createExplanations(model_name='GCN',dataset_name='Cora',type='NC')
createExplanations(model_name='GCN',dataset_name='CiteSeer',type='NC')

createExplanations(model_name='GAT',dataset_name='Cora',type='NC')
createExplanations(model_name='GAT',dataset_name='CiteSeer',type='NC')