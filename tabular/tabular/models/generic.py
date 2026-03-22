import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GenericClassifier:
    def __init__(self, model, verbose, cfg):
        self.model = model
        self.cfg = cfg
        self.verbose = verbose

        self.device = torch.device('cuda' if (torch.cuda.is_available() and cfg['training']['device'] == 'cuda') else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()  # You're welcome to try different losses
        self.optimizer = self._get_optimizer()

    def _get_optimizer(self):
        lr = self.cfg['training']['lr']
        optimizer_name = self.cfg['training']['optimizer']
        if optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        else:
            raise ValueError("Optimizer not supported!")

    def fit(self, x_train, y_train):
        self.model.train()

        if type(x_train) == np.ndarray:
            x_train = torch.from_numpy(x_train).float().to(self.device)
        if type(y_train) == np.ndarray:
            y_train = torch.from_numpy(y_train).long().to(self.device)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        batch_size = self.cfg['training']['batch_size']
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epochs = self.cfg['training']['epochs']
        for epoch in range(epochs):
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
            if self.verbose:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    def predict(self, x_test):
        self.model.eval()
        if type(x_test) == np.ndarray:
            x_test = torch.from_numpy(x_test).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(x_test)
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy() 