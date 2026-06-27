from utils import accuracy
import torch
class Trainer:
    """
    Trainer.fit()

    │
    ├── model.forward()
    │
    ├── criterion.forward()
    │
    ├── criterion.backward()
    │
    ├── model.backward()
    │
    ├── optimizer.update()
    │
    └── Logger
    """
    
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {
            "accuracy" : [],
            "loss" : []
        }
    
    def fit(self, train_loader, max_epochs=100, verbose=10):
        """学習処理
        Parameters
        ----------
        train_loader : torch.util.data.DataLoader的なもの
        max_epochs : 最大epoch数
        verbose : printする間隔
        """
        
        for epoch in range(max_epochs):
            epoch_accuracy  = 0
            epoch_loss = 0
            for x, t in train_loader:
                y, loss = self._train_step(x, t)
                epoch_accuracy  += accuracy(y, t)
                epoch_loss += loss
            
            avg_accuracy = epoch_accuracy  / len(train_loader)
            avg_loss = epoch_loss / len(train_loader)
            self.history["accuracy"].append(avg_accuracy)
            self.history["loss"].append(avg_loss)
    
            if (epoch + 1) % verbose == 0:
                print(f"Epoch [{epoch + 1}/{max_epochs}] Accuracy : {avg_accuracy}% Loss : {avg_loss}")
            
    def _train_step(self, x, t):
        """"1回分の学習処理を実装
        """
        
        y = self.model.forward(x)
        loss = self.criterion.forward(y, t)
        dout = self.criterion.backward()
        self.model.backward(dout)
        self.optimizer.update(self.model.params, self.model.grads)
        
        return y, loss
    
    def evaluate(self, validation_loader):
        pass
    
    def predict(self, test_loader):
        pass