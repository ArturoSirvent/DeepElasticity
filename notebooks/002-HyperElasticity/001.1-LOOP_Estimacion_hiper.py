import os 
BASE_DIR=os.path.abspath("../../")
import sys 
sys.path.append(BASE_DIR)
DATA_DIR="/home/arturosf/Documentos/repos/DeepElasticity/data/002-Neo"
from src.utils.data import DataNeo_2
from src.models import PINN  
from src.train import Trainer_2
import torch 
import matplotlib.pyplot as plt 
import time
from datetime import datetime
import pickle
import numpy as np  
from src.models import PINN_NeoHook


E_values_str=["E10kPa","E15kPa","E20kPa","E25kPa","E30kPa"]
E_values_MPa=[0.01,0.015,0.02,0.025,0.03]


def step_dict_fun_1(net):
    step_dict = {
        "step_1": {"optim": torch.optim.Adam(net.parameters(), lr=1e-2), 
                "epochs": 700},
        "step_2": {"optim": torch.optim.Adam(net.parameters(), lr=1e-3), 
                "epochs": 150},
        "step_3": {"optim": torch.optim.Adam(net.parameters(), lr=1e-4), 
                "epochs": 20}
    }
    return step_dict

def step_dict_fun_2(net):
    step_dict = {
        "step_1": {"optim": torch.optim.Adam(net.parameters(), lr=1e-3), 
                "epochs": 340},
        "step_2": {"optim": torch.optim.Adam(net.parameters(), lr=1e-2), 
                "epochs": 150},
        "step_3": {"optim": torch.optim.Adam(net.parameters(), lr=1e-4), 
                "epochs": 220}
    }
    return step_dict



lista_step_dict_funs=[step_dict_fun_1,step_dict_fun_2]

lame_1_init_values=[10,1,0.1,0.01]
lame_2_init_values=[0.1,0.01,0.001]


for i , (e_val_str, e_val) in enumerate(zip(E_values_str,E_values_MPa)):
    try:
        print(f"Train E value: {e_val_str}")
        data=DataNeo_2("DATOS_HIPERELASTICO_3",5,E=e_val_str,base_dir=DATA_DIR)
        data.load_data(load_stage=3)

        train_init_pos_main,train_disp_main,test_init_pos_main,test_disp_main,position_selected_stresses,return_stress,return_colloc_points=data.prepare_pytorch_data()


        nu=0.4
        E=e_val

        lambda_=E*nu/((1+nu)*(1-2*nu)) #lame1
        mu=E/(2*(1+nu)) #lame2
        print(f"Lambda: {lambda_}",f"mu: {mu}")


        print("Train lame 2")

        #train lame 1 
        for lame1_init in lame_1_init_values:
            print(f"Train lame1: {lame1_init}")
            for j,step_dict_fun in enumerate(lista_step_dict_funs):
                try:
                    print(f"Step: {j}")
                    pinn=PINN_NeoHook([3,25,25,15,3],lame1_init,mu,train_lame2=False,train_lame1=True)
                    step_dict=step_dict_fun(pinn)
                    trainer=Trainer_2(step_dict)
                    trainer.train(pinn,data)

                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.loss_history["PDE"]),label="PDE")
                    plt.plot(np.array(pinn.loss_history["Data"]),label="Data")
                    plt.yscale("log")
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Loss E value: {e_val_str}, Lame1: {lame1_init} Step: {j}")
                    plt.savefig(f"./results/LOSS_E_value_{e_val_str}_lame1_{lame1_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()


                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.params_history["lame1"]),label="BC")
                    plt.axhline(lambda_,color="red")
                    plt.axhline(0,color="grey",linestyle="--")
                    plt.xlabel("Epoch")
                    plt.ylabel("Lame1")
                    plt.title(f"lame1 value: {lambda_}, Lame1: {lame1_init} Step: {j}")
                    plt.savefig(f"./results/PARAMS_E_value_{e_val_str}_lame1_{lame1_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()

                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.params_history["lame1"]),label="BC")
                    plt.axhline(lambda_,color="red")
                    plt.axhline(0,color="grey",linestyle="--")
                    plt.xlabel("Epoch")
                    plt.ylabel("Lame1")
                    plt.title(f"lame1 value: {lambda_}, Lame1: {lame1_init} Step: {j}")
                    plt.ylim(-0.1,0.4)
                    plt.savefig(f"./results/PARAMS_zoom_E_value_{e_val_str}_lame1_{lame1_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()


                    with open(f"./results/pickles/pinn_object_{e_val_str}_lame1_{lame1_init}_step_{j}.pkl","wb") as f:
                        pickle.dump(pinn.loss_history,f)
                except Exception as e:
                    print(f"Error: {e}")

        print("Train lame 2")
        #train lame 2
        for lame2_init in lame_2_init_values:
            print(f"Train lame2: {lame2_init}")
            for j,step_dict_fun in enumerate(lista_step_dict_funs):
                try:
                    print(f"Step: {j}")
                    pinn=PINN_NeoHook([3,25,25,15,3],lambda_,lame2_init,train_lame2=True,train_lame1=False)
                    step_dict=step_dict_fun(pinn)
                    trainer=Trainer_2(step_dict)
                    trainer.train(pinn,data)

                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.loss_history["PDE"]),label="PDE")
                    plt.plot(np.array(pinn.loss_history["Data"]),label="Data")
                    plt.yscale("log")
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Loss E value: {e_val_str}, Lame2: {lame2_init} Step: {j}")
                    plt.savefig(f"./results/LOSS_E_value_{e_val_str}_lame2_{lame2_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()


                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.params_history["lame2"]),label="BC")
                    plt.axhline(mu,color="red")
                    plt.axhline(0,color="grey",linestyle="--")
                    plt.xlabel("Epoch")
                    plt.ylabel("Lame2")
                    plt.title(f"lame2 value: {mu}, Lame2: {lame2_init} Step: {j}")
                    plt.savefig(f"./results/PARAMS_E_value_{e_val_str}_lame2_{lame2_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()

                    plt.figure(figsize=(10,5))
                    plt.plot(np.array(pinn.params_history["lame2"]),label="BC")
                    plt.axhline(mu,color="red")
                    plt.axhline(0,color="grey",linestyle="--")
                    plt.xlabel("Epoch")
                    plt.ylabel("Lame2")
                    plt.title(f"lame2 value: {mu}, Lame2: {lame2_init} Step: {j}")
                    plt.ylim(-0.1,0.4)
                    plt.savefig(f"./results/PARAMS_zoom_E_value_{e_val_str}_lame2_{lame2_init}_step_{j}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    plt.close()



                    with open(f"./results/pickles/pinn_object_{e_val_str}_lame2_{lame2_init}_step_{j}.pkl","wb") as f:
                        pickle.dump(pinn.loss_history,f)
                except Exception as e:
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")
        continue

