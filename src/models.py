# aqui tenemos las diferentes variaciones de los modelos, lo ideal sería aunar en uno con todas 
#las variantes en un PINN etc.  

#librerias

import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks
import numpy as np 

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

# PINN
    

class PINN(DNN):
    '''
        Esta clase hace el registro de todos los parámetros, los cálculos de derivadas y de losses
        Y aplica las restricciones que hagan falta
    '''
    def __init__(self, layers,init_values={"nu":None,"E":None,"alpha":None,"E_ref":None},train_E=True,use_of_alpha=True,device=None,separate_data_losses=True,loss_weights_init={"data":1,"PDE":1,"BC":1}):
        super().__init__(layers)
        
        if device is None:
            self.device= "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device=device
        self.to(self.device)


        #History of losses
        self.loss_history = {"data": [],
                             "PDE": [],
                             "BC": [],
                             "Total":[]}
        
        self.loss_weights_init=loss_weights_init

        self.separate_data_losses=separate_data_losses
        self.params_history ={} #{ "E": [] }
        self.use_of_alpha=use_of_alpha
        self.train_E=train_E
        if self.use_of_alpha and self.train_E:
            #Parameters trials
            self.params_history["E"]= []
            self.params_history["alpha"]= []

            self.alpha = nn.Parameter(torch.tensor(init_values["alpha"],dtype=torch.float32).to(self.device))
            self.E_ref = init_values["E_ref"]
            self.E = torch.tensor((1+self.alpha)*self.E_ref,dtype=torch.float32).to(self.device)
        elif train_E:
            self.params_history["E"]= []

            self.E = nn.Parameter(torch.tensor(init_values["E"],dtype=torch.float32).to(self.device))
        else:
            self.E = torch.tensor(init_values["E"],dtype=torch.float32).to(self.device)
        #inicialización de parametros
        self.nu = init_values["nu"]
        self.loss_function = nn.MSELoss(reduction ='mean')


        #podemos tener dos None, y se equilibran entre ellos, y tambien podemos tener los 3, pero no 1 solo none
        claves_none = [key for key, value in self.loss_weights_init.items() if value is None]
        claves_no_none = [key for key, value in self.loss_weights_init.items() if value is not None]
        self.claves_params=None
        assert len(claves_none)+len(claves_no_none)==3,"Algun error con los pesos"
        if len(claves_none) in [2,3]:
            self.claves_params=[f"w_{i}" for i in claves_none[:(len(claves_none)-1)]]
            #entonces las nones se equilibran entre ellas 
            suma_aux=0
            #para todas las que tiene libre variación
            for clave in self.claves_params:
                setattr(self,clave,nn.Parameter(torch.rand(1,dtype=torch.float32).to(self.device)*(1-suma_aux)))
                suma_aux+=getattr(self, clave).item()
                self.params_history[clave] = []

            #esta dependerá de las otras para que todo sume 1
            self.name_w_updatable=f"w_{claves_none[(len(claves_none)-1)]}"    
            setattr(self, self.name_w_updatable, torch.tensor(1 - suma_aux, dtype=torch.float32).to(self.device))
            self.params_history[self.name_w_updatable] = []
        elif len(claves_none)==0 :
            pass
        else:
            raise Exception("no puede variar solo 1 termino, 2 o 3")

        for clave in claves_no_none:
            setattr(self,f"w_{clave}",self.loss_weights_init[clave])


        #esto es para el closure para el optimizador    
        self.iter_n=0

    def compute_XYZ(self, positions):
        # clone the input data and add AD
        pos = positions.clone().to(self.device)
        X = pos[:,0].reshape(-1,1)
        Y = pos[:,1].reshape(-1,1)
        Z = pos[:,2].reshape(-1,1)
        return X, Y, Z

    def compute_displacements(self,X, Y, Z):
        XYZ = torch.cat((X,Y,Z), dim=1).to( self.device )
        # Compute the output of the DNN
        U = self(XYZ)
        # Separating vector of directional displacements
        u = U[:,0].reshape(-1,1)
        v = U[:,1].reshape(-1,1)
        w = U[:,2].reshape(-1,1)
        return u, v, w

    # Los cálculos de las funciones de pérdida de hacen 1 vez, y luego ya vemos que hacemos
    # con esos datos, si usar pesos, si usar otras cosas...

    def loss_PDE(self, collocation_points, save = True):
        # estos son los puntos en los que imponemos las leyes físicas  
        # puede ser la ecuación de equilibrio, pero también pueden ser otras relaciones
        # que conozcamos entre inputs y outputs, como por ejemplo, relaciones termodinámicas etc..
        # si el output fuera posicion y tensión, entonces podriamos imponer la ley de hooke sobre este
        # input.
        X, Y, Z = self.compute_XYZ(collocation_points)
        u, v, w = self.compute_displacements(X,Y,Z)


        # y ahora tenemos que obtener las derivadas necesarias para aplicar la eq de equilibrio
        epsilon = self.compute_strain(X,Y,Z,u, v, w)
        sigma = self.compute_stress(epsilon)
        div_sigma = self.divergence(sigma,X, Y, Z)

        value_loss_PDE=self.loss_function(div_sigma,torch.zeros_like(div_sigma).to(self.device))
        
        if save:
            self.loss_history["PDE"].append(value_loss_PDE.item())#.to('cpu').detach().numpy())
        
        return value_loss_PDE
    
    def loss_BC(self,pos_reales,sigmas_reales,save=True):
        #aqui tenemos las de Dirichlet y las de Neumann
        # en mi caso que no hay contacto ni nada, pero si tengo nodos fijos!! las de Dirichlet tbn

        #en esta implementación voy a imponer las sigmas solo aqui, en una futura, podría meter también
        #las de Dirichlet para poder darles mas peso

        #calculamos sobre las posiciones de las sigmas que tenemos, las dadas por el modelo
        # predict U
        X,Y,Z=self.compute_XYZ(pos_reales)
        u, v, w=self.compute_displacements(X,Y,Z)
        
        epsilon = self.compute_strain(X,Y,Z,u, v, w)
        sigma = self.compute_stress(epsilon) 
        #este sigma tiene primera dimension batchsize, y segunda dimension 6
        #que corresponde con: s11,s22,s33,s23,s13,s12
        #le imponemos que sean iguales de modo que aplicamos la loss y ya
        value_loss_BC=self.loss_function(sigma,sigmas_reales)

        if save:
            self.loss_history["BC"].append(value_loss_BC.item())#.to('cpu').detach().numpy())

        return value_loss_BC


    def loss_data(self, pos_reales,desp_reales,save=True):
        #pos_reales=pos_reales.to(device)
        u_predict=self(pos_reales)
        if self.separate_data_losses:
            sepatared_loss=torch.nn.MSELoss(reduction="none")
            #aux=self.loss_function(u_predict,desp_reales)
            #esto nos devuelve la diferencia cuadrática de cada elemento, para evaluarlos por
            #separado, vamos a hacer la media en columnas
            # x_mse,y_mse,z_mse=torch.mean(aux,axis=0)
            # x_mse,y_mse,z_mse=torch.sqrt(x_mse),torch.sqrt(y_mse),torch.sqrt(z_mse)
            value_loss_data=torch.mean(torch.sqrt(torch.mean(sepatared_loss(u_predict,desp_reales),axis=0)))
        else:
            value_loss_data=self.loss_function(u_predict,desp_reales)

        if save:
            self.loss_history["data"].append(value_loss_data.item())#.to('cpu').detach().numpy())
        
        return value_loss_data
    
    def loss(self, pos_data,desp_data,pos_colloc,pos_BC,sigmas_BC,save=True):
        #esto hace que se calculen todas las losses de la pinn
        #además, si tenemos un parámetro E, este se actualizará, pe si estamos actualizando alpha en lugar
        #de E, despues de la recalculación tendremos que actualizar E. 

        #actualizamos E antes de calcular nada

        if self.use_of_alpha and self.train_E:
            self.E=(1+self.alpha)*self.E_ref

        # #imponemos que se eviten valores de E negativos 
        # self.E = self.E if self.E.item()>0 else self.E * -1

        #los pesos deben ser positivos
        if self.claves_params:
            for clave in self.claves_params:
                setattr(self, clave, torch.clamp(getattr(self,clave),min=0))

            #hay un peso de los weigts que debe de actualizarse si se están actualizando
            if self.name_w_updatable:
                suma_aux=sum([getattr(self,i).item() for i in self.claves_params])
                setattr(self, self.name_w_updatable, torch.tensor(1 - suma_aux, dtype=torch.float32).to(self.device))




        value_loss_PCE=self.loss_PDE(pos_colloc,save=save)
        value_loss_BC=self.loss_BC(pos_BC,sigmas_BC,save=save)
        value_loss_data=self.loss_data(pos_data,desp_data,save=save)  
        value_loss= self.w_data*value_loss_data + self.w_PDE*value_loss_PCE+ self.w_BC*value_loss_BC

        if save: 
            #self.params_history["nu"].append(self.nu)#self.nu.to('cpu').detach().numpy())
            if self.params_history:
                for key in self.params_history:
                    valor_variable = getattr(self, key).item()#.to('cpu').detach().numpy()
                    self.params_history[key].append(valor_variable)

            self.loss_history["Total"].append(value_loss.item())#.to('cpu').detach().numpy())

        return value_loss

    def compute_gradU(self, X, Y, Z, U, V, W):

        # Compute the gradient of U
        Ux,Uy,Uz = autograd.grad(U, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Uy = autograd.grad(U, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Uz = autograd.grad(U, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of V
        Vx,Vy,Vz = autograd.grad(V, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Vy = autograd.grad(V, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Vz = autograd.grad(V, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of W
        Wx,Wy,Wz = autograd.grad(W, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Wy = autograd.grad(W, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Wz = autograd.grad(W, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        grad_u = torch.cat((Ux , Uy, Uz), dim=1).to(torch.float32)
        grad_v = torch.cat((Vx , Vy, Vz), dim=1).to(torch.float32)
        grad_w = torch.cat((Wx , Wy, Wz), dim=1).to(torch.float32)

        gradU = torch.cat((grad_u, grad_v, grad_w), dim=1).to(torch.float32).reshape(-1,3,3)

        return gradU

    def compute_strain(self, X,Y,Z,u, v, w):
        # Compute strain components using autograd
        nabla_U = self.compute_gradU(X,Y,Z,u, v, w).squeeze()
        strain = 0.5 * (nabla_U + nabla_U.swapaxes(1,2))
        return strain


    def compute_stress(self,strain):
        strain_flat=strain[:,(0,1,2,1,0,0),(0,1,2,2,2,1)]*torch.tensor([1,1,1,2,2,2],dtype=torch.float32).to(self.device)
        #strain_flat=strain[:,(0,1,2,0,0,1),(0,1,2,1,2,2)] # e00,e11,e22,e01,e02,e12
        self.C=(self.E/((1+self.nu)*(1-2*self.nu)))*torch.tensor(
                [[1-self.nu,self.nu,self.nu,0,0,0],
                [self.nu,1-self.nu,self.nu,0,0,0],
                [self.nu,self.nu,1-self.nu,0,0,0],
                [0,0,0,(1-2*self.nu)/2,0,0],
                [0,0,0,0,(1-2*self.nu)/2,0],
                [0,0,0,0,0,(1-2*self.nu)/2]]).float().to(self.device)

        return torch.matmul(self.C,strain_flat.T.float()).T.squeeze() #s11,s22,s33,s23,s13,s12

    def divergence(self,sigma, X, Y, Z):
        div_T = torch.zeros(sigma.shape[0],3).to(self.device)
        div_T[:,0]=autograd.grad(sigma[:,0],X,grad_outputs=torch.ones_like(sigma[:,0]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,5],Y,grad_outputs=torch.ones_like(sigma[:,5]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,4],Z,grad_outputs=torch.ones_like(sigma[:,4]).to(self.device),retain_graph=True)[0].squeeze()
        div_T[:,1]=autograd.grad(sigma[:,5],X,grad_outputs=torch.ones_like(sigma[:,5]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,1],Y,grad_outputs=torch.ones_like(sigma[:,1]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,3],Z,grad_outputs=torch.ones_like(sigma[:,3]).to(self.device),retain_graph=True)[0].squeeze()
        div_T[:,2]=autograd.grad(sigma[:,4],X,grad_outputs=torch.ones_like(sigma[:,4]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,3],Y,grad_outputs=torch.ones_like(sigma[:,3]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,2],Z,grad_outputs=torch.ones_like(sigma[:,2]).to(self.device),retain_graph=True)[0].squeeze()

        return div_T
    

    def step_closure(self,opt,train_init_pos_main2,train_disp_main2,return_colloc_points2,position_selected_stresses2,return_stress2):
        
        opt.zero_grad()
        
        loss = self.loss(train_init_pos_main2,train_disp_main2,return_colloc_points2,position_selected_stresses2,return_stress2)
        
        loss.backward()
        self.iter_n+=1
        # Print material parameters and loss evolition
        print(f'LBFGS iter: {self.iter_n}, Loss: {loss.item()}')
            
        
        return loss



class PINN_mixedForm(DNN):
    '''
        Esta clase hace el registro de todos los parámetros, los cálculos de derivadas y de losses
        Y aplica las restricciones que hagan falta

        Esto será la aprox mixta con el primero el desplazamiento y el segungo la tensión (realmente son 6 componentes).
    '''
    def __init__(self, layers,init_values=None,device=None,separate_data_losses=True,loss_weights={"data":1,"PDE":1,"BC":1,"EC":1}):
        super().__init__(layers)
        
        if device is None:
            self.device= "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device=device
        self.to(self.device)


        #History of losses
        self.loss_history = {"data": [],
                             "PDE": [],
                             "BC": [],
                             "EC":[],
                             "Total":[]}
         #Parameters trials
        self.params_history = { "E": [] ,
                                "alpha": []}

        #inicialización de parametros
        self.nu = init_values[0]
        self.alpha = nn.Parameter(torch.tensor(init_values[2],dtype=torch.float32).to(self.device))
        self.E_ref = init_values[1]
        self.E = torch.tensor((1+self.alpha)*self.E_ref,dtype=torch.float32).to(self.device)
        self.separate_data_losses=separate_data_losses
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.w_data=loss_weights["data"]
        self.w_PDE=loss_weights["PDE"]
        self.w_BC=loss_weights["BC"]
        self.w_EC=loss_weights["EC"]


        self.iter_n=0

    def compute_XYZ(self, positions):
        # clone the input data and add AD
        pos = positions.clone().to(self.device)
        X = pos[:,0].reshape(-1,1)
        Y = pos[:,1].reshape(-1,1)
        Z = pos[:,2].reshape(-1,1)
        return X, Y, Z

    def compute_displacements(self,X, Y, Z):
        XYZ = torch.cat((X,Y,Z), dim=1).to( self.device )
        # Compute the output of the DNN
        U = self(XYZ)
        # Separating vector of directional displacements
        u = U[:,0].reshape(-1,1)
        v = U[:,1].reshape(-1,1)
        w = U[:,2].reshape(-1,1)
        return u, v, w, U[:,3:]
    # Los cálculos de las funciones de pérdida de hacen 1 vez, y luego ya vemos que hacemos
    # con esos datos, si usar pesos, si usar otras cosas...

    def loss_PDE(self, collocation_points, save = True):
        # estos son los puntos en los que imponemos las leyes físicas  
        # puede ser la ecuación de equilibrio, pero también pueden ser otras relaciones
        # que conozcamos entre inputs y outputs, como por ejemplo, relaciones termodinámicas etc..
        # si el output fuera posicion y tensión, entonces podriamos imponer la ley de hooke sobre este
        # input.
        collocation_points=collocation_points.to(self.device)
        X, Y, Z = self.compute_XYZ(collocation_points)
        u, v, w,_ = self.compute_displacements(X,Y,Z)


        # y ahora tenemos que obtener las derivadas necesarias para aplicar la eq de equilibrio
        epsilon = self.compute_strain(X,Y,Z,u, v, w)
        sigma = self.compute_stress(epsilon)
        div_sigma = self.divergence(sigma,X, Y, Z)

        value_loss_PDE=self.loss_function(div_sigma,torch.zeros_like(div_sigma).to(self.device))
        
        if save:
            self.loss_history["PDE"].append(value_loss_PDE.item())#.to('cpu').detach().numpy())
        
        return value_loss_PDE
    
    def loss_BC(self,pos_reales,sigmas_reales,save=True):
        #aqui tenemos las de Dirichlet y las de Neumann
        # en mi caso que no hay contacto ni nada, pero si tengo nodos fijos!! las de Dirichlet tbn

        #en esta implementación voy a imponer las sigmas solo aqui, en una futura, podría meter también
        #las de Dirichlet para poder darles mas peso

        #calculamos sobre las posiciones de las sigmas que tenemos, las dadas por el modelo
        # predict U
        X,Y,Z=self.compute_XYZ(pos_reales)
        u, v, w,_=self.compute_displacements(X,Y,Z)
        
        epsilon = self.compute_strain(X,Y,Z,u, v, w)
        sigma = self.compute_stress(epsilon) 
        #este sigma tiene primera dimension batchsize, y segunda dimension 6
        #que corresponde con: s11,s22,s33,s23,s13,s12
        #le imponemos que sean iguales de modo que aplicamos la loss y ya
        value_loss_BC=self.loss_function(sigma,sigmas_reales.to(self.device))

        if save:
            self.loss_history["BC"].append(value_loss_BC.item())#.to('cpu').detach().numpy())

        return value_loss_BC

    #def loss_DBC(fix_nodes)
    #   pass

    def loss_data(self, pos_reales,desp_reales,save=True):
        desp_reales=desp_reales.to(self.device)
        u_predict=self(pos_reales.to(self.device))[:,:3]
        if self.separate_data_losses:
            sepatared_loss=torch.nn.MSELoss(reduction="none")
            #aux=self.loss_function(u_predict,desp_reales)
            #esto nos devuelve la diferencia cuadrática de cada elemento, para evaluarlos por
            #separado, vamos a hacer la media en columnas
            # x_mse,y_mse,z_mse=torch.mean(aux,axis=0)
            # x_mse,y_mse,z_mse=torch.sqrt(x_mse),torch.sqrt(y_mse),torch.sqrt(z_mse)
            value_loss_data=torch.mean(torch.sqrt(torch.mean(sepatared_loss(u_predict,desp_reales),axis=0)))
        else:
            value_loss_data=self.loss_function(u_predict,desp_reales)

        if save:
            self.loss_history["data"].append(value_loss_data.item())#.to('cpu').detach().numpy())
        
        return value_loss_data
    

    def loss_ConstEq(self, pos_colloc,save=True):

        X,Y,Z=self.compute_XYZ(pos_colloc)
        u, v, w,sigma_pred=self.compute_displacements(X,Y,Z)
        
        epsilon = self.compute_strain(X,Y,Z,u, v, w)
        sigma = self.compute_stress(epsilon) #s11,s22,s33,s23,s13,s12  
        value_loss_ConstEq=self.loss_function(sigma,sigma_pred)
        if save:
            self.loss_history["EC"].append(value_loss_ConstEq.item())#.to('cpu').detach().numpy())

        return value_loss_ConstEq

    
    def loss(self, pos_data,desp_data,pos_colloc,pos_BC,sigmas_BC,save=True):
        #esto hace que se calculen todas las losses de la pinn
        #además, si tenemos un parámetro E, este se actualizará, pe si estamos actualizando alpha en lugar
        #de E, despues de la recalculación tendremos que actualizar E. 

        #actualizamos E antes de calcular nada

        self.E=(1+self.alpha)*self.E_ref

        value_loss_PCE=self.loss_PDE(pos_colloc,save=save)
        value_loss_BC=self.loss_BC(pos_BC,sigmas_BC,save=save)
        value_loss_data=self.loss_data(pos_data,desp_data,save=save) 
        value_loss_ConstEq=self.loss_ConstEq(pos_colloc,save=save)

        value_loss= self.w_data*value_loss_data + self.w_PDE*value_loss_PCE+ self.w_BC*value_loss_BC + self.w_EC*value_loss_ConstEq


        if save: 
            #self.params_history["nu"].append(self.nu)#self.nu.to('cpu').detach().numpy())
            self.params_history["E"].append(self.E.to('cpu').detach().numpy())
            self.params_history["alpha"].append(self.alpha.to('cpu').detach().numpy())
            self.loss_history["Total"].append(value_loss.item())#.to('cpu').detach().numpy())

        return value_loss

    def compute_gradU(self, X, Y, Z, U, V, W):

        # Compute the gradient of U
        Ux,Uy,Uz = autograd.grad(U, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Uy = autograd.grad(U, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Uz = autograd.grad(U, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of V
        Vx,Vy,Vz = autograd.grad(V, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Vy = autograd.grad(V, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Vz = autograd.grad(V, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of W
        Wx,Wy,Wz = autograd.grad(W, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Wy = autograd.grad(W, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Wz = autograd.grad(W, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        grad_u = torch.cat((Ux , Uy, Uz), dim=1).to(torch.float32)
        grad_v = torch.cat((Vx , Vy, Vz), dim=1).to(torch.float32)
        grad_w = torch.cat((Wx , Wy, Wz), dim=1).to(torch.float32)

        gradU = torch.cat((grad_u, grad_v, grad_w), dim=1).to(torch.float32).reshape(-1,3,3)

        return gradU

    def compute_strain(self, X,Y,Z,u, v, w):
        # Compute strain components using autograd
        nabla_U = self.compute_gradU(X,Y,Z,u, v, w).squeeze()
        strain = 0.5 * (nabla_U + nabla_U.swapaxes(1,2))
        return strain


    def compute_stress(self,strain):
        strain_flat=strain[:,(0,1,2,1,0,0),(0,1,2,2,2,1)]*torch.tensor([1,1,1,2,2,2],dtype=torch.float32).to(self.device)
        #strain_flat=strain[:,(0,1,2,0,0,1),(0,1,2,1,2,2)] # e00,e11,e22,e01,e02,e12
        self.C=(self.E/((1+self.nu)*(1-2*self.nu)))*torch.tensor(
                [[1-self.nu,self.nu,self.nu,0,0,0],
                [self.nu,1-self.nu,self.nu,0,0,0],
                [self.nu,self.nu,1-self.nu,0,0,0],
                [0,0,0,(1-2*self.nu)/2,0,0],
                [0,0,0,0,(1-2*self.nu)/2,0],
                [0,0,0,0,0,(1-2*self.nu)/2]]).float().to(self.device)

        return torch.matmul(self.C,strain_flat.T.float()).T.squeeze() #s11,s22,s33,s23,s13,s12

    def divergence(self,sigma, X, Y, Z):
        div_T = torch.zeros(sigma.shape[0],3).to(self.device)
        div_T[:,0]=autograd.grad(sigma[:,0],X,grad_outputs=torch.ones_like(sigma[:,0]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,5],Y,grad_outputs=torch.ones_like(sigma[:,5]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,4],Z,grad_outputs=torch.ones_like(sigma[:,4]).to(self.device),retain_graph=True)[0].squeeze()
        div_T[:,1]=autograd.grad(sigma[:,5],X,grad_outputs=torch.ones_like(sigma[:,5]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,1],Y,grad_outputs=torch.ones_like(sigma[:,1]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,3],Z,grad_outputs=torch.ones_like(sigma[:,3]).to(self.device),retain_graph=True)[0].squeeze()
        div_T[:,2]=autograd.grad(sigma[:,4],X,grad_outputs=torch.ones_like(sigma[:,4]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,3],Y,grad_outputs=torch.ones_like(sigma[:,3]).to(self.device),retain_graph=True)[0].squeeze()+autograd.grad(sigma[:,2],Z,grad_outputs=torch.ones_like(sigma[:,2]).to(self.device),retain_graph=True)[0].squeeze()

        return div_T
    

    def step_closure(self,opt,train_init_pos_main2,train_disp_main2,return_colloc_points2,position_selected_stresses2,return_stress2):
        
        opt.zero_grad()
        
        loss = self.loss(train_init_pos_main2,train_disp_main2,return_colloc_points2,position_selected_stresses2,return_stress2)
        
        loss.backward()
        self.iter_n+=1
        # Print material parameters and loss evolition
        print(f'LBFGS iter: {self.iter_n}, Loss: {loss.item()}')
            
        
        return loss
    


# modelo hiperelastico Neo-Hookeano   
class PINN_NeoHook(DNN):
    def __init__(self, layers,init_lame1,init_lame2):
        super().__init__(layers)
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)    

        'Define loss function'
        self.loss_function = torch.nn.MSELoss(reduction ='mean')
        
        'Initialize iterator'
        self.iter = 0
        
        'Initialize our new parameter mu and bulk as tensor (Inverse problem)' 
        self.lame1 =torch.tensor([float(init_lame1)]).to(self.device)# nn.Parameter(torch.tensor([float(init_lame1)], requires_grad=True).float().to(device))
        self.lame2 = nn.Parameter(torch.tensor([float(init_lame2)], requires_grad=True).float().to(self.device))
        #self.density = torch.tensor([float(rho)], requires_grad=True).float()
        
        'History of losses'
        self.loss_history = {"Data": [],
                             "PDE": [],
                             "symmetry": [],
                             "BC": [],
                             "Total":[]}
        'Parameters trials'
        self.params_history = {"lame2": [] } #"lame1": [],
                                
        
    
    def loss_data(self,pos_real, despl_real, save = False):
        u_nn = self(pos_real)
        loss = self.loss_function(u_nn, despl_real)

        save and self.loss_history["Data"].append(loss.to('cpu').detach().numpy()) 
        
        return loss
        
    def loss_physics(self, pos_f, save = False):
        pos_f=pos_f.to(torch.float32)
        num_points = pos_f.shape[0]
        
        eyes = torch.eye(3).repeat(num_points,1,1).to(self.device)

        X,Y,Z = self.compute_XYZ(pos_f)

        u, v, w = self.compute_displacements(X,Y,Z)

        F = self.compute_F(u, v, w, X, Y, Z, eye=eyes)

        C = self.compute_C(F)

        S = self.compute_S(C, material="Neo-Hookean-FEBIO", eye=eyes)

        P = self.compute_P(F,S)

        div_P = self.compute_div_P(P, X, Y, Z)        

        # Computing the mass force tensor
        # massForce = compute_mass_force(self.density, num_points)

        # Computing Res = 0
        eq_residual = div_P #+ massForce

        # Res_hat is just an auxiliar term to copmute the loss (is zero)
        eq_residual_ground_truth = torch.zeros_like(eq_residual).to(self.device)

        # Computing the Physics Loss
        loss_equilibrium = self.loss_function(eq_residual, eq_residual_ground_truth)

        loss_symmetry = self.compute_loss_symmetry(P, F)

        save and self.loss_history["symmetry"].append(loss_symmetry.to('cpu').detach().numpy())
        save and self.loss_history["PDE"].append(loss_equilibrium.to('cpu').detach().numpy()) 

        return loss_equilibrium, loss_symmetry
    
    def compute_gradU(self, X, Y, Z, U, V, W):

        # Compute the gradient of U
        Ux,Uy,Uz = autograd.grad(U, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Uy = autograd.grad(U, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Uz = autograd.grad(U, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of V
        Vx,Vy,Vz = autograd.grad(V, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Vy = autograd.grad(V, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Vz = autograd.grad(V, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of W
        Wx,Wy,Wz = autograd.grad(W, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)
        # Wy = autograd.grad(W, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        # Wz = autograd.grad(W, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        grad_u = torch.cat((Ux , Uy, Uz), dim=1).to(torch.float32)
        grad_v = torch.cat((Vx , Vy, Vz), dim=1).to(torch.float32)
        grad_w = torch.cat((Wx , Wy, Wz), dim=1).to(torch.float32)

        gradU = torch.cat((grad_u, grad_v, grad_w), dim=1).to(torch.float32).reshape(-1,3,3)

        return gradU
    
    def compute_strain(self, X,Y,Z,u, v, w):
        # Compute strain components using autograd
        nabla_U = self.compute_gradU(X,Y,Z,u, v, w).squeeze()
        strain = 0.5 * (nabla_U + nabla_U.swapaxes(1,2))
        return strain
    
    # def loss_BC(self,pos_reales,sigmas_reales,save=True):
    #         #aqui tenemos las de Dirichlet y las de Neumann
    #         # en mi caso que no hay contacto ni nada, pero si tengo nodos fijos!! las de Dirichlet tbn

    #         #en esta implementación voy a imponer las sigmas solo aqui, en una futura, podría meter también
    #         #las de Dirichlet para poder darles mas peso

    #         #calculamos sobre las posiciones de las sigmas que tenemos, las dadas por el modelo
    #         # predict U
    #         X,Y,Z=self.compute_XYZ(pos_reales)
    #         u, v, w,_=self.compute_displacements(X,Y,Z)
            
    #         epsilon = self.compute_strain(X,Y,Z,u, v, w)
    #         sigma = self.compute_stress(epsilon) 
    #         #este sigma tiene primera dimension batchsize, y segunda dimension 6
    #         #que corresponde con: s11,s22,s33,s23,s13,s12
    #         #le imponemos que sean iguales de modo que aplicamos la loss y ya
    #         value_loss_BC=self.loss_function(sigma,sigmas_reales.to(self.device))

    #         if save:
    #             self.loss_history["BC"].append(value_loss_BC.item())#.to('cpu').detach().numpy())

    #         return value_loss_BC


    def compute_XYZ(self, positions):
        # clone the input data and add AD
        pos = positions.clone().to(self.device)
        X = pos[:,0].reshape(-1,1).to(self.device)
        
        Y = pos[:,1].reshape(-1,1).to(self.device)
        
        Z = pos[:,2].reshape(-1,1).to(self.device)
        
        return X, Y, Z

    def compute_displacements(self, X, Y, Z):
        XYZ = torch.cat((X,Y,Z), dim=1).to(self.device)

        # Compute the output of the DNN
        U = self(XYZ)            

        # Separating vector of directional displacements
        u = U[:,0].reshape(-1,1).to(self.device)
        v = U[:,1].reshape(-1,1).to(self.device)
        w = U[:,2].reshape(-1,1).to(self.device)

        return u, v, w

    def compute_F(self, u, v, w, X, Y, Z, eye=None):
        if eye is None:
            eye = torch.eye(3).repeat(X.shape[0],1,1).to(self.device)
        # Compute the gradient of U
        Ux = autograd.grad(u, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Uy = autograd.grad(u, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Uz = autograd.grad(u, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        # Compute the gradient of V
        Vx = autograd.grad(v, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Vy = autograd.grad(v, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Vz = autograd.grad(v, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        
        # Compute the gradient of W
        Wx = autograd.grad(w, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Wy = autograd.grad(w, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        Wz = autograd.grad(w, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        
        grad_u = torch.cat((Ux , Uy, Uz), dim=1).to(torch.float32)
        grad_v = torch.cat((Vx , Vy, Vz), dim=1).to(torch.float32)
        grad_w = torch.cat((Wx , Wy, Wz), dim=1).to(torch.float32)

        gradU = torch.cat((grad_u, grad_v, grad_w), dim=1).to(torch.float32).reshape(-1,3,3)
        
        F = eye + gradU

        return F

    def compute_C(self, F):
        # Compute the right Cauchy-Green tensor
        C = torch.matmul(F.transpose(1,2), F)
        return C
        
    def compute_J(self, C):
        # Compute the determinant of C
        detC = torch.linalg.det(C)
        J = torch.sqrt(detC)
        return J
    

    def compute_S(self, C, material, eye ):
        C=C.cpu()
        J = self.compute_J(C)

        invC = torch.linalg.inv(C).to(self.device)

        num_points = C.shape[0]

        # Compute the second Piola-Kirchhoff tensor
        if material == 'Neo-Hookean-ONSAS':
            num_points = C.shape[0]
            JTerm = (J*(J-1)).reshape(num_points,1,1)
            S = self.mu*(eye - invC ) + self.bulk * JTerm * invC 
        elif material == "Neo-Hookean-FEBIO":
            S = self.lame1 * (eye - invC) + self.lame2 * torch.log(J).reshape(num_points,1,1).to(self.device) * invC

        elif material == 'Mooney-Rivlin':
            S = self.mu * (C - eye) + self.bulk * (J - eye) * invC + self.kappa * (J - eye)**2 * invC
        
        return S.to(self.device)

    def compute_P(self, F, S):
        # Compute the first Piola-Kirchhoff tensor
        P  = torch.matmul(F.cpu(),S.cpu())
        return P.to(self.device)

    def compute_div_P(self, P, X, Y, Z):

        
        ## Computing derivative of each component of P        
        P11 = P[:,0,0].reshape(-1,1)
        P12 = P[:,0,1].reshape(-1,1)
        P13 = P[:,0,2].reshape(-1,1)
        P11x = autograd.grad(P11, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P12y = autograd.grad(P12, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P13z = autograd.grad(P13, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        P21 = P[:,1,0].reshape(-1,1)
        P22 = P[:,1,1].reshape(-1,1)
        P23 = P[:,1,2].reshape(-1,1)
        P21x = autograd.grad(P21, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P22y = autograd.grad(P22, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P23z = autograd.grad(P23, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        P31 = P[:,2,0].reshape(-1,1)
        P32 = P[:,2,1].reshape(-1,1)
        P33 = P[:,2,2].reshape(-1,1)
        P31x = autograd.grad(P31, X, torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P32y = autograd.grad(P32, Y, torch.ones([Y.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]
        P33z = autograd.grad(P33, Z, torch.ones([Z.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)[0]

        ## Adding derivatives to compone the divergence of each vector
        divP1 = P11x + P12y + P13z
        divP2 = P21x + P22y + P23z
        divP3 = P31x + P32y + P33z

        ## Concatenating the divergence of each vector
        divP = torch.cat((divP1, divP2, divP3), dim=1).to(torch.float32)        
        return divP
    
    def compute_loss_symmetry(self, F, P):
        # symmetry of the tensor FtP
        F_transpose = torch.transpose(F, 1, 2)
        P_transpose = torch.transpose(P, 1, 2)
        PFt = torch.matmul(P, F_transpose)
        FPt = torch.matmul(F, P_transpose)
        # Computing the loss of the symmetry of the tensor FtP
        loss_symmetry = self.loss_function(PFt, FPt)
        return loss_symmetry
    

    def loss(self,  pos_real, despl_real, pos_f, save = False):
        loss_d  = self.loss_data(pos_real,despl_real, save)
        loss_physics, loss_symmetry  = self.loss_physics(pos_f, save)
       
        # weights should sum 1
        loss_val = loss_d + loss_physics + loss_symmetry 
        if save:
            #self.params_history["lame1"].append(self.lame1.to('cpu').detach().numpy())
            self.params_history["lame2"].append(self.lame2.to('cpu').detach().numpy())
            self.loss_history["Total"].append(loss_val.to('cpu').detach().numpy())

        return loss_val

    # def step_closure(self, dataset=breast_dataset):
        
    #     optimizer.zero_grad()
        
    #     loss = self.loss(dataset, True)
        
    #     loss.backward()
        
    #     self.iter += 1

    #     if True:
    #         print(
    #             'Optim iter: %d, Loss: %.6f' %
    #             (   
    #                 self.iter,
    #                 loss.cpu().detach().numpy(),
    #             )
    #         )
        
    #     return loss
    
    def compute_dif_euclidean_distances(self, dataset):
        "Computes the difference between the displacement field of the neural network and the ground truth"

        # Extract validation dataset
        dataloader = dataset.validation_dataloader
        batch_size = dataloader.batch_size

        # Compute [d_nn[0] - d_ground_truths[0], d_nn[1] - d_ground_truths[1], d_nn[2] - d_ground_truths[2]]
        # [u_nn - u_gt, v_nn - v_gt, w_nn - w_gt] for each point
        # Sotres the subtraction of the displacement field of the neural network and the ground truth
        euclidean_distances = np.zeros((len(dataset.validation_dataset), 3))
        
        for idx, data in enumerate(dataloader):
            data = data.to(self.device)
            data.requires_grad = False

            d_nn = self(data[:, dataset.position_indexes])
            d_ground_truths = data[:, dataset.displacement_indexes] 
            d_diff = (d_nn -  d_ground_truths).to('cpu').detach().numpy()

            indexes_to_fill_min, indexes_to_fill_max = idx * batch_size, (idx + 1) * batch_size
            euclidean_distances[indexes_to_fill_min : indexes_to_fill_max] = d_diff
            
        norm_euclidean_distances = np.linalg.norm(euclidean_distances, axis=1)

        return euclidean_distances, norm_euclidean_distances

    def compute_euclidean_mean_error(self, dataset):
        "Computes MEE = mean(||d_nn - d_gt||)"

        euclidean_distances, norm_euclidean_distances = self.compute_dif_euclidean_distances(dataset)

        mean_euclidean_error = np.mean(norm_euclidean_distances, axis=0)

        return mean_euclidean_error, euclidean_distances
    
    def compute_relative_euclidean_mean_error(self, dataset):
        "Computes MREE = mean(||d_nn - d_gt|| / ||d_gt||)"

        # ||d_nn - d_gt|| 
        _, norm_euclidean_distances = self.compute_dif_euclidean_distances(dataset)

        # ||d_gt||
        d_gt = dataset.validation_dataset[:][:, dataset.displacement_indexes]
        norm_d_gt = np.linalg.norm(d_gt, axis=1)

        # Find indexes where norm is not zero 
        not_null_disp_indexes = np.where(norm_d_gt != 0)[0]

        relative_euclidean_error = norm_euclidean_distances[not_null_disp_indexes] / norm_d_gt[not_null_disp_indexes]
        mean_relative_euclidean_error = np.mean(relative_euclidean_error, axis=0)

        return mean_relative_euclidean_error, relative_euclidean_error
    

    def compute_dif_euclidean_positions(self, dataset):
        """
        Computes the difference between the deformed positions field of the the neural network and the ground truth.
        ||x_def_nn - x_def_gt|| at each point
        """

        # Extract validation dataset
        dataloader = dataset.validation_dataloader
        batch_size = dataloader.batch_size

        # Compute [d_nn[0] - d_ground_truths[0], d_nn[1] - d_ground_truths[1], d_nn[2] - d_ground_truths[2]]
        # [u_nn - u_gt, v_nn - v_gt, w_nn - w_gt] for each point
        # Sotres the subtraction of the displacement field of the neural network and the ground truth
        dif_euclidean_positions = np.zeros((len(dataset.validation_dataset), 3))
        
        for idx, data in enumerate(dataloader):
            data = data.to(self.device)
            data.requires_grad = False

            # Compute displacements
            d_nn = self(data[:, dataset.position_indexes])
            d_ground_truths = data[:, dataset.displacement_indexes] 

            # Reference position
            X0 = data[:, dataset.position_indexes]
            # Compute deformed positions
            X_def_nn = X0 + d_nn
            X_def_gt = X0 + d_ground_truths
            
            # [x_def_nn - x_def_gt, y_def_nn - y_def_gt, z_def_nn - z_def_gt] for each point
            X_diff = (X_def_nn -  X_def_gt).to('cpu').detach().numpy()

            indexes_to_fill_min, indexes_to_fill_max = idx * batch_size, (idx + 1) * batch_size
            dif_euclidean_positions[indexes_to_fill_min : indexes_to_fill_max] = X_diff
            
            # TODO: add the abis in the euclidean positions

        return dif_euclidean_positions

    def compute_mean_absolute_error(self, dataset):
        "Computes MAE = mean(|x_def_nn - x_def_gt|), mean(|y_def_nn - y_def_gt|), mean(|z_def_nn - z_def_gt|)"

        dif_euclidean_positions = self.compute_dif_euclidean_positions(dataset)

        mean_absolute_error = np.mean(np.abs(dif_euclidean_positions), axis=0)
        return mean_absolute_error, dif_euclidean_positions

