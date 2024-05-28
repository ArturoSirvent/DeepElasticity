# UPV - Prueba1 Prostata Lineal   
19:05 | 2023-05-30 (Tuesday)   
Type: #atomic   
Main: #status_unasigned  
Tags: #DeepLearning     
Relates: [[UPV - PINNS]], [[Physics - CauchyStressTensor]]
Status:  #status_toDo  

---
# About
Básicamente es hacer esto con la próstata. Los datos que me pasará serán de una deformación nodal sobre una sola geometría.  


---
# Contenido

## Ecuación diferencial
Por un lado, es la deformacion y los desplazamientos los que nos definen las dinámicas del sistema, y luego tenemos la relación entre deformaciones y tensiones, que es lo que nos caracteriza el material y tal.  

> [!done]- Duda
> **Duda: ¿Que es realmente el modelo, donde entra en juego estricatamente el tipo de material, que es de la teoria básica y donde empieza el tema lineal, lineal es el materia o el modelo?**  
> Esfuerzo == tensiones ??? Sí, y el tensor de esfuerzo de cauchy es el estandar tbn. 
> 

La ecuación diferencial que introducimos en nuestra red, parte de las derivadas de los desplazamientos, eso nos da las deformaciones:  
![[Pasted image 20230531232059.png]]  

Pero con esto no vamos a ningún lado, necesitamos algo que igualar a 0, necesitamos algo que lo relacione con el material, una pérdida necesitamos.  
La tensiones son lo que nos relaciona con el material, de forma lineal con las deformaciones:  

![[Pasted image 20230531232335.png]]  


Los parámetros mu y E son del material.  

Por último, usamos ecuaciones de conservación o de equilibrio, para indicarle a la red un PDE que debe cumplir:    
![[Pasted image 20230531232451.png]]


Y por último en [wikipedia](<https://es.wikipedia.org/wiki/Elasticidad_(mec%C3%A1nica_de_s%C3%B3lidos)>) nos pode esta sección que ya no me he leido bien, pero parece algo a considerar tambien:  
![[Pasted image 20230531232646.png]]  



## Preguntas 7Junio2023  
- ¿Por qué hay 1674 nodos para los stresses, pero tenemos unos 4k nodos?  
- Los que tienen limitaciones sobre ciertas direcciones, no pasa nada porque tengan esfuerzos apliacados no ?
- La gran limitación ahora es que tenemos aplicado en los nodos una fuerza en Newtons, pero esa fuerza debería ser fuerza por masa, de modo que o tenemos la densidad o tenemos el area de aplicación o algo.    
- La parte de la divergencia okey, pero es el otro término el de la fuerza el que supone una duda.   



> [!done] solved !!  
> La simulación tiene limitaciones en el movimiento de unos nodos, limitando el movimiento en una sola dirección, yo había pensado que son los que se desplazan por el transductor, pero resulta que la fuerza que ha aplicado para la simulación es al contrario:  
> ![[Pasted image 20230611192214.png]]  
> Los azules son los que reciben un fuerza, y los morados los que tienen la limitación.  
> **Vale nada, se está limitando de esa forma ahí porque hay hueso y se limita el desplazamiento en esa dirección**
> 

- Hay un -1 que no entiendo :
```python
normalized_X_tensor[:,col] = (X_tensor[:,col]-min_X_col)*2/(max_X_col-min_X_col) -1 #para que es este -1?
```




## Sigo con las ecuaciones diferenciales del equilibrio  
[[A general solution of equations of equilibrium in linear elasticity.pdf]]    
![[Pasted image 20230612222216.png]]  
- https://en.wikiversity.org/wiki/Elasticity/Constitutive_relations   
- https://en.wikipedia.org/wiki/Linear_elasticity 
- https://personalpages.manchester.ac.uk/staff/matthias.heil/Lectures/Elasticity/Material/Chapter5.pdf  
![[Pasted image 20230612205851.png]]   
from [here](http://web.mit.edu/16.20/homepage/3_Constitutive/Constitutive_files/module_3_with_solutions.pdf).    
- https://sbrisard.github.io/posts/20140112-elastic_constants_of_an_isotropic_material-03.html#:~:text=In%20isotropic%20elasticity%2C%20the%20stiffness,combination%20of%20J%20and%20K.  
![[Pasted image 20230612210911.png]]    
De [wiki](https://en.wikipedia.org/wiki/Linear_elasticity).    


Investigando como implementar la ecuación del equilibrio y tal, necesitaba usar el hessiano de la función, pero no me deja calcular el hessiano de una red neuronal muy compleja, lo debo hacer con autograd. Sin embargo aqui propone una solución con una capacidad reciente de pytorch: https://stackoverflow.com/questions/71471406/how-to-compute-the-hessian-of-a-large-neural-network-in-pytorch  , la solución viene de [aquí](https://github.com/pytorch/pytorch/issues/49171#issuecomment-933814662)  


De [aqui](https://dnicolasespinoza.github.io/node16.html) sacamos las relaciones constitutivas:  
![[Pasted image 20230613200543.png]]

Estoy raiao, porque lo que sale en el código del plano lineal es esto:  
![[Pasted image 20230614143713.png|800]]


## Derivar las ecuaciones del plano lineal  
Parto de las ecuaciones de la segunda ley de newton:   
![[Pasted image 20230614175451.png]]  

De esta ecuación, estamos suponiendo el caso estático, luego los desplazamientos no cambian.  
Y las fuerzas $F$, las estamos imponiendo en la parte de BC de la ecuación diferencial, luego es algo que estamos sumando igualmente a la loss. Parece que no está del todo correcto, pero tampoco mal.  

La relación entre el estres y las deformaciones es la ecuacion constitutiva, la cual se define por el material.  Para materiales elásticos, lineales.  
![[Pasted image 20230614175705.png]]
Tendremos unas relaciones que vienen dadas por el tensor de stiffness.  Esto al final es la ley de Hooke y nada mas:  
![[Pasted image 20230614175852.png]]  


Donde E es el parametro de young, y mu el de poisson. 

Todo esto es basicamente la [ley de hooke en 3d](https://en.m.wikipedia.org/wiki/Hooke%27s_law)  

Cuando aproximamos para el caso 2D, tenemos que hacer una de dos posibles aproximaciones: Tensión plana o deformación plana.   
Aquí se explica genial -> https://ingenieriabasica.es/teoria-de-la-elasticidad-deformacion-y-tension-plana/   
En el caso que hicieron, es el de Tensión plana, donde la deformación resulta no ser nula, de modo que $\epsilon_{33}$ no es nula y la tenemos que poner en función de las otras, porque no podemos poner $\frac{\partial w}{\partial z}$, en su lugar, ponemos epsion en función de las otras dos.    

Bueno aun asi, no me sale como puede ser posible que no haya un $\frac{\partial^2 u}{\partial^2 y}$

Maria Jose me pasa este desarrollo ([[Nota 15 jun 2023.pdf]]) y medice por correo:  

> [!NOTE]
> [10:10, 15/6/2023] Maria jose PINNS UPV: Hola. He estado haciendo el cálculo para la hipótesis de tensión plana que es la que se puede aplicar a un plano y he encontrado los coeficientes sin suponer que la deformación z es cero. Sin embargo, sigo pensando que hay un error en las derivadas.   
> [10:11, 15/6/2023] Maria jose PINNS UPV: Te he enviado 2 correos, no le hagas caso al primero que el documento tiene un error.   
> [10:11, 15/6/2023] Maria jose PINNS UPV: Es el segundo el que está bien deducido. Aún así, échale un vistazo no sea que me haya equivocado en algo.

Para el caso 2D con la aproximación de **Tension Plana** $\sigma_z=0$, terminamos con esta eq:  
![[Pasted image 20230617164212.png]]  

Y la otra *supongo* que será exactamente igual pero con los u cambiados por v y viceversa, y los x por y y viceversa.  



## Resultados para 12JUL

### Lo he dejado aqui:  
- https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
- https://en.wikipedia.org/wiki/Hooke%27s_law  
- https://www.continuummechanics.org/greenstrain.html
![[Pasted image 20230711144037.png]]  
![[Pasted image 20230711144350.png]]  
![[Pasted image 20230711144952.png]]


1. He corregido lo del stress, ahora se calcula la BC sobre todo el tensor.  
2. Calculamos epsilon de forma vectorial.  
3. Entrenamos para encontrar los parámetros E y mu. 
4. He probado poniendo pesos a las losses, y no esta claro, deberá ser sobre la variación de estas loses.

![[Pasted image 20230712150500.png]]
![[Pasted image 20230712150515.png]]  

![[Pasted image 20230712150527.png]]  


Para el ajuste de valores :  
![[Pasted image 20230712150608.png]]  

![[Pasted image 20230712150626.png]]   
![[Pasted image 20230712152916.png]]

Otras veces tenía oscilaciones un poco raras que se veían en las losses y en los parametros:  
![[Pasted image 20230712153449.png]]  
![[Pasted image 20230712153457.png]]  

Subiéndole un poco el peso a los datos reales, obtengo algo un poco mejor:    
![[Pasted image 20230712154023.png]]
![[Pasted image 20230712154057.png]]  


---
# References
Tengo que hacer lo que hacen aquí pero con la próstata. Un poco más complex.  -> https://github.com/CDiazCuadro/ElasticityPINN/blob/main/examples/LinearElasticPlate/PlaneStress/PINN_LinearElasticPlate.ipynb   (recordar invitarle a comer <3)  
La PDE que usan aquí es :  
```python
    def loss_PDE(self, pos_f, save = False):
                       
        # clone the input data and add AD
       
        pos = pos_f.clone()
        pos.requires_grad = True

        # predict u
        U = self.dnn(pos)

        # compute the derivatives togheter
        dU = autograd.grad(U, pos, torch.ones([pos.shape[0], 2]).to(device),retain_graph=True, create_graph=True)[0]

        u_x = dU[:,0].reshape(-1,1)
        v_y = dU[:,1].reshape(-1,1)

        # compute second derivatives
        ddU = autograd.grad(dU, pos, torch.ones([pos.shape[0], 2]).to(device),retain_graph=True, create_graph=True)[0]

        u_xx = ddU[:,0].reshape(-1,1)
        v_yy = ddU[:,1].reshape(-1,1)

        # Shift columns of du tensor
        dU_shifted = torch.roll(dU, 1, 1)

        cross_ddU = autograd.grad(dU_shifted, pos, torch.ones([pos.shape[0], 2]).to(device),retain_graph=True, create_graph=True)[0]

        u_xy = cross_ddU[:,1].reshape(-1,1)
        v_yx = cross_ddU[:,0].reshape(-1,1)

        # PDE f = 0
        f = [ (2*G)/(1-nu) * (u_xx + nu*v_yx) + G*(u_xy+ v_yy) ,  
              G*(u_xx + v_yx) + (2*G)/(1-nu)*(v_yy + nu*u_xy) ]
        f_tensor= torch.cat((f[0] , f[1]), dim=1).to(torch.float32)

        
        # f_hat is just an auxiliar term to copmute the loss (is zero)
        loss_f = self.loss_function(f_tensor, f_hat)

        save and self.loss_history["PDE"].append(loss_f.to('cpu').detach().numpy()) 
        return loss_f
```  

