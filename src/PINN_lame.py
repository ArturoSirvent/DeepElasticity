#librerias

import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks

from src.DNN import DNN


class PINN_lame(DNN):
    '''
    Esta clase hace el registro de todos los parámetros, los cálculos de derivadas y de losses
    Y aplica las restricciones que hagan falta
    '''
    def __init__(self, layers, init_values={"lambda": None, "mu": None}, train_lambda=True,train_mu=True, device=None, separate_data_losses=True, loss_weights_init={"data": 1, "PDE": 1, "BC": 1}):
        super().__init__(layers)
        
        # Establecer el dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device is None else device
        self.to(self.device)

        # Historial de pérdidas
        self.loss_history = {"data": [], "PDE": [], "BC": [], "Total": []}
        self.loss_weights_init = loss_weights_init
        self.separate_data_losses = separate_data_losses
        self.params_history ={}

        # Inicialización de las constantes de Lamé
        if train_lambda:
            self.lambda_lame = nn.Parameter(torch.tensor(init_values["lambda"], dtype=torch.float32).to(self.device)) 
            self.params_history["lambda_lame"]= []

        else:
            self.lambda_lame =torch.tensor(init_values["lambda"], dtype=torch.float32).to(self.device)

        if train_mu:
            self.mu_lame = nn.Parameter(torch.tensor(init_values["mu"], dtype=torch.float32).to(self.device))
            self.params_history["mu_lame"]= []
        else:
            self.mu_lame = torch.tensor(init_values["mu"], dtype=torch.float32).to(self.device)

        # Función de pérdida
        self.loss_function = nn.MSELoss(reduction='mean')


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
            if self.params_history:
                for key in self.params_history:
                    valor_variable = getattr(self, key).item()#.to('cpu').detach().numpy()
                    print(key,valor_variable)
                    self.params_history[key].append(valor_variable)

            self.loss_history["Total"].append(value_loss.item())#.to('cpu').detach().numpy())

        return value_loss

    def compute_gradU(self, X, Y, Z, U, V, W):

        # Compute the gradient of U
        Ux,Uy,Uz = autograd.grad(U, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)

        # Compute the gradient of V
        Vx,Vy,Vz = autograd.grad(V, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)

        # Compute the gradient of W
        Wx,Wy,Wz = autograd.grad(W, (X,Y,Z), torch.ones([X.shape[0], 1]).to(self.device),retain_graph=True, create_graph=True)

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

        C = torch.tensor(
            [[self.lambda_lame + 2 * self.mu_lame, self.lambda_lame, self.lambda_lame, 0, 0, 0],
             [self.lambda_lame, self.lambda_lame + 2 * self.mu_lame, self.lambda_lame, 0, 0, 0],
             [self.lambda_lame, self.lambda_lame, self.lambda_lame + 2 * self.mu_lame, 0, 0, 0],
             [0, 0, 0, self.mu_lame, 0, 0],
             [0, 0, 0, 0, self.mu_lame, 0],
             [0, 0, 0, 0, 0, self.mu_lame]]).float().to(self.device)


        return torch.matmul(C,strain_flat.T.float()).T.squeeze() #s11,s22,s33,s23,s13,s12

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