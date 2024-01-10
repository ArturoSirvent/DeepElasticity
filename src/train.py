from abc import ABC,abstractmethod
from dataclasses import dataclass
import torch
from functools import partial

class AbstractTrainer(ABC):

    @abstractmethod
    def train():
        raise NotImplementedError
    

# @dataclass
# class Step:
#     epocs: int =100
#     optimizer: torch.optim.Optimizer = torch.optim.Adam



class Trainer(AbstractTrainer):
    def __init__(self,steps_dict) -> None:
        self.steps_dict=steps_dict
        
    def train(self,model,data):
        #aqui definimos un bucle que recorrerá los steps uno detras de otro 
        if not data._data_loaded:
            try:
                data.load_data()
            except Exception as e:
                raise e("no se ha cargado la data")
        
        train_init_pos_main,train_disp_main,test_init_pos_main,test_disp_main,position_selected_stresses,return_stress,return_colloc_points=data.prepare_pytorch_data()
        for step_name in self.steps_dict:
            step=self.steps_dict[step_name]
            optimizer=step["optim"]
            #cada step tiene unas cosas unos pasos definidos 
            if isinstance(optimizer,torch.optim.LBFGS):

                optimizer.step(partial(model.step_closure,optimizer,train_init_pos_main,train_disp_main,return_colloc_points,position_selected_stresses,return_stress))

            else:  
                for epoch in range(step["epochs"]): 
                    optimizer.zero_grad()
                    loss=model.loss(train_init_pos_main,train_disp_main,return_colloc_points,position_selected_stresses,return_stress)
                    loss.backward()
                    optimizer.step()

                    print("Epoch: ", epoch, "loss: ", loss.item())




