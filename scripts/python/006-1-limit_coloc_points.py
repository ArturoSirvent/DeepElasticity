#librerias

import torch

import pandas as pd
import numpy as np
import re 
import random
from dataclasses import dataclass
import plotly.express as px
from scipy.spatial import ConvexHull, Delaunay
import os 
BASE_DIR=os.path.abspath("../../")
import sys 
sys.path.append(BASE_DIR)
DATA_DIR=os.path.join(BASE_DIR,"data/001-LinearElasticity")
from src.models import PINN  
from src.train import Trainer
import torch 
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt


#vamos a hacer una función para cargar todos los datos de una   
@dataclass
class Data:
    folder: str
    load_stage: int
    base_dir:str = "/home/arturo/Documents/programacion_stuff/DeepElasticity/data/001-LinearElasticity"
    E: str = None
    #===== not seteables ====
    initialPosition_data_pd: pd.DataFrame =None
    stress_data_pd: pd.DataFrame =None
    displacement_data_pd: pd.DataFrame=None
    restricted_data_pd: pd.DataFrame =None
    force_data_pd: pd.DataFrame =None 
    final_data_pd: pd.DataFrame =None
    collocation_data_np:pd.DataFrame =None
    Pos_min: np.ndarray = None
    Pos_max: np.ndarray = None
    adjancet_matrix: np.ndarray=None
    device : str ="cpu"
    _data_loaded : bool =False
    _is_normaliced: bool = False

    @staticmethod
    def keep_line(line):
        line_split=line.split()
        try:
            int(line_split[0])
            #si no dio error, okey entra
            return True
        except:
            return False

    def load_data(self,load_stage=10,normalize_pos=False):
        # carga las posiciones de los nodos    
        initialPosition_data_path = f"{self.base_dir}/ARTURO_TEST_1/NODES.txt"

        with open(initialPosition_data_path,"r") as f:
            aux_list=f.readlines()

        initialPosition_data=[i for i in aux_list if self.keep_line(i)]
        initialPosition_data=[i.strip("\n") for i in initialPosition_data]
        initialPosition_data=[i.split() for i in initialPosition_data]
        initialPosition_data_np=np.array(initialPosition_data).astype(float)[:,[0,1,2,3]]
        self.initialPosition_data_pd=pd.DataFrame(initialPosition_data_np,columns=["Node","X","Y","Z"]).set_index("Node")
        self.Pos_min=self.initialPosition_data_pd.min().to_numpy()
        self.Pos_max=self.initialPosition_data_pd.max().to_numpy()

        #noramlizamos las posiciones 
        if normalize_pos:
            self._is_normaliced=True
            #la normalizacion de hace min max porque queremos escalar los datos no cambiar su distribucion espacial
            self.initialPosition_data_pd=(self.initialPosition_data_pd-self.initialPosition_data_pd.min())/(self.initialPosition_data_pd.max()-self.initialPosition_data_pd.min())
            #ahora han cambiado los valores de las posiciones minima y máxima.
            self.Pos_min=self.initialPosition_data_pd.min().to_numpy()
            self.Pos_max=self.initialPosition_data_pd.max().to_numpy()
        #carga el streess
        if self.E is not None:
            stress_data_path = f"{self.base_dir}/{self.folder}/PSOL_{load_stage}_NODAL_STRESSES_E{self.E}.txt"
        else: 
            stress_data_path = f"{self.base_dir}/{self.folder}/PSOL_{load_stage}_NODAL_STRESSES.txt"

        with open(stress_data_path,"r") as f:
            aux_list=f.readlines()

        stress_data=[i for i in aux_list if self.keep_line(i)]
        stress_data=[i.strip("\n").strip() for i in stress_data]
        patron = "[-.\d]+E-*\d{3}|^\d{0,4}|0\.0000"

        stress_data=[re.findall(patron,i) for i in stress_data]
        stress_data=np.array(stress_data,dtype=float)
        self.stress_data_pd=pd.DataFrame(stress_data,columns=["Node","SX","SY","SZ","SXY","SYZ","SXZ"]).set_index("Node")[["SX","SY","SZ","SYZ","SXZ","SXY"]] # lo queremos asi : s11,s22,s33,s23,s13,s12

        # cargar datos de desplazamiento  
        if self.E is not None:
            displacement_data_path = f"{self.base_dir}/{self.folder}/PSOL_{load_stage}_NODAL_DISP_E{self.E}.txt"
        else:
            displacement_data_path = f"{self.base_dir}/{self.folder}/PSOL_{load_stage}_NODAL_DISP.txt"

        with open(displacement_data_path,"r") as f:
            aux_list=f.readlines()


        displacement_data=[i for i in aux_list if self.keep_line(i)]
        displacement_data=[i.strip("\n").strip() for i in displacement_data]
        patron = r"[-.\d]+E-*\d{3}|^\d{0,4}|-?\d+\.\d+|0\.0000"

        displacement_data=[re.findall(patron,i) for i in displacement_data]
        displacement_data=np.array(displacement_data,dtype=float)
        self.displacement_data_pd=pd.DataFrame(displacement_data,columns=["Node","UX","UY","UZ","USUM"]).set_index("Node")

        # cargar datos de boundaries en el movimiento
        #cargar los nodos fijos
        with open(f"{self.base_dir}/ARTURO_TEST_1/RESTRINGED_NODES.txt","r") as f:
            restricted_data=f.readlines()

        restricted_data=[i for i in restricted_data if self.keep_line(i)]
        restricted_data=[i.strip("\n") for i in restricted_data]
        restricted_data=[i.split() for i in restricted_data]
        restricted_data_np=np.array(restricted_data)[:,[0,1]]
        self.restricted_data_pd=pd.DataFrame(restricted_data_np,columns=["Node","Direccion"])
        self.restricted_data_pd=self.restricted_data_pd.groupby("Node")["Direccion"].apply(lambda x : list(x)).to_frame().sort_index()
        self.restricted_data_pd.index=self.restricted_data_pd.index.astype(int)
        self.restricted_data_pd=self.restricted_data_pd.sort_index()
        self.restricted_data_pd=self.restricted_data_pd.rename(columns={"Direccion":"Restricciones"})

        with open(f"{self.base_dir}/ARTURO_TEST_1/FORCE_ON_NODES.txt","r") as f:
            force_data=f.readlines()


        force_data=[i for i in force_data if self.keep_line(i)]
        force_data=[i.strip("\n") for i in force_data]
        force_data=[i.split() for i in force_data]
        force_data_np=np.array(force_data)[:,[0,1,2]]
        self.force_data_pd=pd.DataFrame(force_data_np,columns=["Node","Direccion_Fuerza","Fuerza"])
        self.force_data_pd["Fuerza"]=self.force_data_pd["Fuerza"].astype(float)
        self.force_data_pd=self.force_data_pd.set_index("Node")
        self.force_data_pd.index=self.force_data_pd.index.astype(int)


        #ponemos todos los datos en común usando los nodos como clave   
        self.final_data_pd=self.initialPosition_data_pd.merge(self.stress_data_pd,left_index=True,right_index=True,how="left").merge(self.force_data_pd,left_index=True,right_index=True,how="left").merge(self.displacement_data_pd,left_index=True,right_index=True,how="left").merge(self.restricted_data_pd,left_index=True,right_index=True,how="left")

        self.final_data_pd["Final_X"]=self.final_data_pd["X"]-self.final_data_pd["UX"]
        self.final_data_pd["Final_Y"]=self.final_data_pd["Y"]-self.final_data_pd["UY"]
        self.final_data_pd["Final_Z"]=self.final_data_pd["Z"]-self.final_data_pd["UZ"]
        
        self._data_loaded=True

    def create_colloc_points(self,n_colloc_in=70000,factor_de_escala=1.05):
        #generamos puntos aleatorios dentro de un cubo grande, pero tenemos que definir la envolvente mediante 
        #un convex hull y luego ya podemos generar los puntos aleatorios y filtrarlos para que esten dentro de la envolvente
        envolvente = ConvexHull(self.initialPosition_data_pd)
        centroide = np.mean(envolvente.points[envolvente.vertices], axis=0)
        factor_de_escala = factor_de_escala
        puntos_vertices_escalados = np.array([centroide + (punto - centroide) * factor_de_escala for punto in envolvente.points[envolvente.vertices]])
        delaunay = Delaunay(puntos_vertices_escalados)
        num_puntos_aleatorios = n_colloc_in
        #puntos_aleatorios = np.random.rand(num_puntos_aleatorios, 3)
    
        puntos_aleatorios = torch.tensor(np.random.uniform(0, self.Pos_max, size=(num_puntos_aleatorios, 3)))

        indices_dentro = delaunay.find_simplex(puntos_aleatorios) >= 0
        puntos_dentro = puntos_aleatorios[indices_dentro]
        self.collocation_data_np=puntos_dentro.reshape(-1, 3)
        
    def prepare_pytorch_data(self,n_colloc_in=70000,percentage_stress_data=0.8,train_percent=0.8):
        """
        para el entrenamiento necesitamos diferentes conjuntos de datos

        1. los datos experimentales que saldrán de los datos sin limitaciones de movimiento
        pero no haremos ninguna diferenciación más, los que tengan una fuerza aplicada nos da igual
        lo tomamos también

        1.1. Otro set de datos que son los que tienen limitaciones totales del movimiento
        1.2. Otro que será los que tengan limitaciones direccionales del movimiento

        estos dos anteriores, simplemente tienen desplamiento nulo en las direcciones que correspondan
        por eso lo vamos a meter con los datos normales, pero si quisieramos darles mayor importancia
        podríamos tenerlos en un término a parte den la funcion de perdida.   

        2. Los collocation points, esto son x,y,z repartidas por todo el dominio que nos interese  


        3. Las BC. Aquí entran las NBC, y en el futuro la DBC de antes. Antes imponíamos solo donde estaba
        aplicada la fuerza, pero esto no tiene porque ser así, es más, estamos muy limitados si así es. Si solo quisiéramos 
        darle información de la Fuerza, habría que ver otra manera creo yo. Por ahora, le voy a dar todos o un subconjunto de los 
        puntos y valores de sigma en la superficie, para que lo tenga como referencia para aprender la fisica y hacer bien los
        desplazamientos.

        Los datos de test será sacados del conjunto de datos no restringidos de desplazamientos

        El tema de la normalizacion: ----
        """

        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        # colloc points
        if self.collocation_data_np is None:
            self.create_colloc_points(n_colloc_in=n_colloc_in)
        return_colloc_points=torch.tensor(self.collocation_data_np,requires_grad=True)

        # desp_data, tanto los limitados como el resto  

        # no limitados, separamos en 80 y 20
        indx_non_restricted=[int(i) for i in self.displacement_data_pd.index if i not in self.restricted_data_pd.index ]
        random.shuffle(indx_non_restricted)
        self.indx_train_non_restricted,self.indx_test_non_restricted = indx_non_restricted[:int(len(indx_non_restricted)//(1/train_percent))], indx_non_restricted[int(len(indx_non_restricted)//(1/train_percent)):]
        
        train_init_pos_non_restricted, train_disp_non_restricted = torch.tensor(self.initialPosition_data_pd.loc[self.indx_train_non_restricted,["X","Y","Z"]].to_numpy(),requires_grad=True) , torch.tensor(self.displacement_data_pd.loc[self.indx_train_non_restricted,["UX","UY","UZ"]].to_numpy(),requires_grad=True)
        test_init_pos_non_restricted, test_disp_non_restricted = torch.tensor(self.initialPosition_data_pd.loc[self.indx_test_non_restricted,["X","Y","Z"]].to_numpy(),requires_grad=True) , torch.tensor(self.displacement_data_pd.loc[self.indx_test_non_restricted,["UX","UY","UZ"]].to_numpy(),requires_grad=True)

        # limitados
        initpos_restricted_data=torch.tensor(self.initialPosition_data_pd.loc[self.restricted_data_pd.index,["X","Y","Z"]].to_numpy(),requires_grad=True)

        disp_restricted_data=torch.tensor(self.displacement_data_pd.loc[self.restricted_data_pd.index,["UX","UY","UZ"]].to_numpy(),requires_grad=True)

        train_init_pos_main,train_disp_main=torch.concat([train_init_pos_non_restricted,initpos_restricted_data]),torch.concat([train_disp_non_restricted,disp_restricted_data])
        test_init_pos_main,test_disp_main=test_init_pos_non_restricted, test_disp_non_restricted

        #los indices los tenemos trazados:   

        self.index_train=list(self.indx_train_non_restricted) + self.restricted_data_pd.index.to_list() 
        self.index_test=list(self.indx_test_non_restricted)
        # data del NCB
        # voy a escoder un pocentaje de datos de estos de los que dar.
        selected_stresses=self.stress_data_pd.sample(frac=percentage_stress_data,axis=0)
        self.index_stress=selected_stresses.index.to_list()
        position_selected_stresses=torch.tensor(self.initialPosition_data_pd.loc[selected_stresses.index].to_numpy(),requires_grad=True)
        return_stress=torch.tensor(selected_stresses.to_numpy(),requires_grad=True)
        # si quisieramos hacer una separacion
        #from sklearn... import train_test_split
        #train_test_split(self.stress_data_pd.to_numpy())

        return train_init_pos_main.float().to(self.device),train_disp_main.float().to(self.device),test_init_pos_main.float().to(self.device),test_disp_main.float().to(self.device),position_selected_stresses.float().to(self.device),return_stress.float().to(self.device),return_colloc_points.float().to(self.device)

E_real_str="0.020"
DATA_DIR="/home/arturo/Documents/programacion_stuff/DeepElasticity/data/001-LinearElasticity"
RESULTS_DIR="/home/arturo/Documents/programacion_stuff/DeepElasticity/informes/sin_norm_resultadosMarzo2024_V2" 
os.makedirs(RESULTS_DIR,exist_ok=True)
data=Data("MULTIPLE_E_VALUES",10,E=E_real_str,base_dir=DATA_DIR)
data.load_data(load_stage=10,normalize_pos=False)
train_init_pos_main,train_disp_main,test_init_pos_main,test_disp_main,position_selected_stresses,return_stress,return_colloc_points=data.prepare_pytorch_data(n_colloc_in=80000)

EPOCHS1=4800
EPOCHS2=1200
EPOCHS3=700
pinn_arch=[3,40,40,40,40,40,40,40,3]

list_param_1=[0.01,1,1e2,1e4,1e6]
list_param_3=[1,0.1,0.01]

for param1 in list_param_1:
    for param2 in list_param_1:
        for param3 in list_param_3:
            try:
                init_values={"nu":.4,"E":param3}
                loss_weights_init={"data":1,"PDE":param1,"BC":param2}

                pinn=PINN(pinn_arch,use_of_alpha=False,train_E=True,init_values=init_values,loss_weights_init=loss_weights_init)
                step_dict = {
                    "step_1": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-3), 
                            "epochs": EPOCHS1},
                    "step_2": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-4), 
                            "epochs": EPOCHS2},
                    "step_3": {"optim": torch.optim.Adam(pinn.parameters(), lr=1e-5), 
                            "epochs": EPOCHS3}
                }
                trainer=Trainer(step_dict)
                trainer.train(pinn,data)

                plt.figure()

                for i in pinn.loss_history.keys():
                    if i in loss_weights_init.keys():
                        aux_w=loss_weights_init[i]
                    else:
                        aux_w=1.0
                    plt.plot(np.array(pinn.loss_history[i])*aux_w,label=i)
                    plt.yscale("log")
                plt.legend()
                plt.tight_layout() 
                plt.savefig(f"{RESULTS_DIR}/losses_{param1}_{param2}_{param3}.png")

                plt.figure()
                plt.plot(pinn.params_history["E"])
                plt.axhline(y=0.02,color="red")      
                plt.tight_layout()
                plt.savefig(f"{RESULTS_DIR}/E_{param1}_{param2}_{param3}.png")
            except Exception as e:
                print(e)
                continue
