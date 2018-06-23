import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(
        self,input_size,batch_size, hidden_size,embed_dim,num_layers):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        super(MyNet, self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout = 0.2, bidirectional=True,batch_first = True)
        self.fc = nn.Linear(hidden_size*2,embed_dim*input_size)

    def forward(self, x):
       #x is N*T*F tensor
       hidden = (
                torch.randn(self.num_layers*2, self.batch_size, self.hidden_size),
                torch.randn(self.num_layers*2, self.batch_size, self.hidden_size)
                )
       out, hidden = self.lstm(x,hidden)
       vec = self.fc(out)
       return vec

def loss_func(out,target,embed_dim):
    #out is N*T*(F*embed_dim)
    #reshape it to -1*embed_dim
    out = out.view(-1,embed_dim)
    target = target.view(-1,3)
    A = torch.matmul(target, torch.t(target))
    A_ = torch.matmul(out, torch.t(out))
    return torch.norm((A - A_),p=2)

#TODO
#Add batch norm module at the first (or all) Module
#define the loss function in the paper
#Extract the feature and seg it to batch*100*513
#Train on 100 frame segs then 400 segs

embed_dim = 6
inputs = torch.randn(10,100, 11)
target = torch.randn(10,1100, 3)
model = MyNet(11,10,30,embed_dim,2)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
for _ in range(100):
    out = model(inputs)
    optimizer.zero_grad()
    loss = loss_func(out,target,embed_dim)
    print(loss)
    loss.backward()
    optimizer.step()
