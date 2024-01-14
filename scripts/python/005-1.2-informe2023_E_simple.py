import os 
BASE_DIR=os.path.abspath("../../")
import sys 
sys.path.append(BASE_DIR)
DATA_DIR=os.path.join(BASE_DIR,"data/001-LinearElasticity")
from src.utils.data import Data
from src.models import PINN  
from src.train import Trainer
import torch 
import time
from datetime import datetime
import pickle



EPOCHS1=200
EPOCHS2=1600
EPOCHS3=1200
fecha_actual_str = datetime.now().strftime("%Y-%m-%d")

RESULTADOS_DIR=os.path.join(BASE_DIR,"informes/endYear2023/resultados")

DAILY_RESULTADOS_DIR=os.path.join(RESULTADOS_DIR,fecha_actual_str)
os.makedirs(DAILY_RESULTADOS_DIR,exist_ok=True)

E_options=["0.020","0.040","0.050","0.060","0.005","0.009"]#"0.020",
E_inits=[0.01,0.07,0.1,0.3,1]
loss_weights_init={"data":1e1,"PDE":1e5,"BC":1e6}
pinn_arch=[3,80,80,80,80,80,3]

historial_E_train={}
for i,E_real_str in enumerate(E_options):
    print(f"E_real: {E_real_str}")
    data=Data("MULTIPLE_E_VALUES",10,E=E_real_str,base_dir=DATA_DIR)
    data.load_data()
    train_init_pos_main,train_disp_main,test_init_pos_main,test_disp_main,position_selected_stresses,return_stress,return_colloc_points=data.prepare_pytorch_data()
    dict_E_inits={}
    for E_init in E_inits:
        print(f"\t E_init: {E_init}")

        time_init=time.time()
        init_values={"nu":.4,"E":E_init}
        pinn=PINN(pinn_arch,use_of_alpha=False,init_values=init_values,loss_weights_init=loss_weights_init)
        step_dict = {
            "step_1": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-2), 
                    "epochs": EPOCHS1},
            "step_2": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-3), 
                    "epochs": EPOCHS2},
            "step_3": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-4), 
                    "epochs": EPOCHS3}
            }
        trainer=Trainer(step_dict)
        trainer.train(pinn,data)
        time_fin=time.time()
        dict_E_inits[str(E_init)]={"model_object":pinn,"time":time_fin-time_init,"E_real":float(E_real_str)}
    historial_E_train[E_real_str]=dict_E_inits
    del dict_E_inits
    #guardamos 
    fecha_hora_actual_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    archivo_pickle=os.path.join(DAILY_RESULTADOS_DIR, f'{fecha_hora_actual_str}.pkl')
    with open(archivo_pickle, 'wb') as archivo:
        pickle.dump(historial_E_train, archivo)

