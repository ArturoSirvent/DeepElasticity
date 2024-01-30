#librerias

import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks


# Red neuronal  

class DNN(nn.Module):
    def __init__(self,layers,init_w="xavier",normalize_inputs=False):
        super().__init__() 
        self.layers=layers
        self.normalize_inputs=normalize_inputs
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
    
        if init_w=="xavier":
            #Xavier Normal Initialization
            for i in range(len(layers)-1):
                nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                
                # set biases to zero
                nn.init.zeros_(self.linears[i].bias.data)
        # elif init_w=="gorot":
        #         nn.init. (self.linears[i].weight.data, gain=1.0)
                
        #         # set biases to zero
        #         nn.init.zeros_(self.linears[i].bias.data)


    def forward(self,x):
              
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)             
        # if self.normalize_inputs:
        #     xn = normalize(x, Pos_min, Pos_max)

        # convert to float
        a = x.float()
        

        # inpunt and hidden layers forward computation
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)          
            a = self.activation(z)

        # output layer forward computation            
        a = self.linears[-1](a)
        
        return a