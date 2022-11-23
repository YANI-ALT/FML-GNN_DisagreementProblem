import torch
from torch_geometric.loader import DataLoader

class TrainingNC():
    def  __init__(self,model,data,lr=0.005,weight_decay=5e-4):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data=data
        self.model=model

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(self.data.x, self.data.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(self,mask):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[mask] == self.data.y[mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return acc

class TrainingGC():
    def  __init__(self,model,dataset,lr=0.01):
        self.dataset=dataset
        self.model=model
        self.test_loader=None
        self.train_loader=None
        # self.create_dataloader()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()

        for data in self.train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            if len(data.y.shape)>1:
                loss = self.criterion(out, data.y[:,0].type(torch.LongTensor))  # Compute the loss.
            else:
                loss = self.criterion(out, data.y)  # Compute the loss.
            #  print(loss.item())
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
        
    def test(self,loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
        
        

def TrainModel(model,data,params,type):
    final_output={}
    if(type=='NC'):
        training_setup=TrainingNC(model,data,params['lr'],params['weight_decay'])
        epochs=params['epochs']
        
        for epoch in range(1, epochs):
            loss = training_setup.train_step()
            val_acc = training_setup.test(data.val_mask)
            test_acc = training_setup.test(data.test_mask)
            if params['verbose'] and epoch%10==0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            
            if epoch==epochs-1:
                final_output['Loss']=loss
                final_output['Val']=val_acc
                final_output['Test']=test_acc
    elif type=='GC':
        training_setup=TrainingGC(model,data,params['lr'])
        torch.manual_seed(12345)
        dataset = data.shuffle()
        split=int(0.6*len(data))
        train_dataset = dataset[:split]
        test_dataset = dataset[split:]

        train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader=DataLoader(test_dataset, batch_size=64, shuffle=False)
        training_setup.train_loader=train_loader
        training_setup.test_loader=test_loader
        epochs=params['epochs']
             
        for epoch in range(1, epochs):
            training_setup.train()
            train_acc = training_setup.test(train_loader)
            test_acc = training_setup.test(test_loader)
            if epoch==epochs-1:
                final_output['Loss']=None
                final_output['Train']=train_acc
                final_output['Test']=test_acc
            if params['verbose'] and epoch%10==0:
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    

    if params['save_wts']!='':
        print("Saving model in {}".format(params['save_wts']))
        torch.save(training_setup.model.state_dict(), params['save_wts'])

    return final_output