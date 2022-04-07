from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch
import pytorch_lightning as pl


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class plNet(pl.LightningModule):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output, learning_rate = 1e-5):
        super(plNet, self).__init__()
        self.learning_rate = learning_rate
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)  # output layer
        self.w1 = torch.nn.Parameter(torch.ones(n_feature))
        self.w2 = torch.nn.Parameter(torch.ones(n_feature))

    def forward(self, x1, x2):
        x = torch.mul(x1, self.w1) + torch.mul(x2, self.w2)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        p = self.forward(x1, x2)
        loss = F.mse_loss(p, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        p = self.forward(x1, x2)
        loss = F.mse_loss(p, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # print(get_lr(optimizer))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=50)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode="min")
        # scheduler = {"scheduler": lr_scheduler, "reduce_on_plateau": True, "monitor": "val_loss", "patience": 5, "verbose": True}
        # return [optimizer], [scheduler]
        return optimizer


class TrDataset(torch.utils.data.Dataset):
    """ Information about the datasets """
    def __init__(self, X1, X2, Y):
        self.x1 = X1
        self.x2 = X2
        self.y = Y

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        return len(self.x1)



if __name__ == "__main__":
    model = plNet(10, 3, 3, 3)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print( name, param.data)
    x1 = torch.Tensor(2, 10)
    x2 = torch.Tensor(2, 10)
    fedfwd = model(x1, x2)

    w1 = torch.tensor([1,1,1,1,1,2,2,2,2,2])
    tmp = torch.mul(x1, w1)
    print(tmp.shape)