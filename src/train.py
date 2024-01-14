from abc import ABC,abstractmethod
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
        epoch=0
        for step_name in self.steps_dict:
            print(step_name)
            step=self.steps_dict[step_name]
            optimizer=step["optim"]
            #cada step tiene unas cosas unos pasos definidos 
            if isinstance(optimizer,torch.optim.LBFGS):

                optimizer.step(partial(model.step_closure,optimizer,train_init_pos_main,train_disp_main,return_colloc_points,position_selected_stresses,return_stress))

            else:  
                for _ in range(step["epochs"]): 
                    optimizer.zero_grad()
                    loss=model.loss(train_init_pos_main,train_disp_main,return_colloc_points,position_selected_stresses,return_stress)
                    loss.backward()
                    optimizer.step()
                    epoch+=1
                    print("Epoch: ", epoch, "loss: ", loss.item())


# train_dict={
#     "E_inits":[]

# }

# def es_lista_de_listas(lista):
#     for elemento in lista:
#         if not isinstance(elemento, list):
#             return False
#     return True

# class LoopTrainer:
#     def __init__(self,model,train_dict) -> None:
#         #este metodo va a recorrer muchos datasets, dando muchos valores iniciales, y probando siempre el mismo modelo
#         # train_dict tiene toda la informacion del entrenamiento, las epocas, los valores iniciales a probar
#         # los valores reales a probar
#         self.historico_train={}
#         self.model=model
#         self.train_dict=train_dict
#         self.E_init=self.train_dict["E_init"]





#     def loop_train(self):
#         #el loop se hace de cada valor real, para todos los valores iniciales

#         for i,E in enumerate(self.E_options):
#             #para cada valor de opciones ponemos todos los valores de init E
#             if es_lista_de_listas(self.E_init):
#                 E_init_aux=self.E_init[i]
#             else:
#                 E_init_aux=self.E_init

#             #cargamos los datos    

#             hist_W_inits={}
#             for E_init in E_init_aux:
#                 # para cada elemento de estos hacemos el entrenamiento. 
                 
                 
