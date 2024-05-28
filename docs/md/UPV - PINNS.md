# UPV - PINNS  
2023-04-30 (Sunday) | 12:41   
Type: #notebook  
Main: [[Main - UPV]]
Tags: #UPV  #DeepLearning  
Relates: [[Topic - PINNS]], [[Topic - Physics]]

---
# About 
Esto va dedicado a lo de PINNS más relativos al propósito de la UPV que es Ingeniería de Materiales.  **Pero por el momento va a ser equivalente a [[Topic - PINNS]]**


---
# Historia / Trabajos relacionados    
Esto tiene como objetivo proporcionar una agrupación para hacer un texto introductorio sobre trabajos relacionados en PINNs en este informe: [[UPV - InformeGeneralitat2023]]

**por ahora esta en medeley**

---
# Teoría
[The Crunch Group – The collaborative research work of George Em Karniadakis](https://sites.brown.edu/crunch-group/)

> [!NOTE] De [wiki](https://en.wikipedia.org/wiki/Physics-informed_neural_networks):
> PINNs can be designed to solve two classes of problems:
> 
> -   data-driven solution
> -   data-driven discovery



> [!note] Soft and hard form/problem
> Sobre el tema debil-fuerte. [recurso1](https://www.youtube.com/watch?v=Z-FnP2myvKw&ab_channel=MachineLearning%26Simulation)  [recurso2](https://www.youtube.com/watch?v=Z-FnP2myvKw&ab_channel=MachineLearning%26Simulation) 
> Cuando resolvemos un problema con ecuaciones diferenciales en forma debil o fuerte, lo que estamos haciendo es imponer ciertas restricciones en los puntos, o en regiones (debil). La segunda es mucho menos restrictiva que la primera, y puedes ser muy conveniente a veces cuando no se encuentra la la solución.  
> En principio estábamos aplicando condiciones fuertes, en los puntos se quiere cumplir. Este es un [paper](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.7176) donde se aplican condicciones débiles.  


### Variantes
- [[DL - XPINNs]]
- [[DL - cPINNs]]-> conservative PINNs  
- [CAN-PINN](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001906): A fast physics-informed neural network based on coupled-automatic–numerical differentiation method
- [B-PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0021999120306872): Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data

### Physics-constrained machine learning 
Esto es casi una variantes, pero no llega a ser [[Topic - PINNS]] ni [[ML - Physics & ML]]
- Physics-constrained machine learning for scientific computing -> [amazon post](https://www.amazon.science/blog/physics-constrained-machine-learning-for-scientific-computing)
- [Guiding continuous operator learning through physics-based boundary constraints](https://www.amazon.science/publications/guiding-continuous-operator-learning-through-physics-based-boundary-constraints)


## Ideas para el aprox  
Check:  
- https://arxiv.org/abs/2308.02467  
- 🌟 https://arxiv.org/abs/2211.09373

### GNN   

[[DL - GraphNeuralNetworks]]
Se habla mucho de del Mesage passing. 
[Graph Element Networks: adaptive, structured computation and memory](https://arxiv.org/pdf/1904.09019.pdf) 
- https://onlinelibrary.wiley.com/doi/epdf/10.1002/pamm.202200306  

#### Papers del tema GNN   
- https://arxiv.org/pdf/2205.08332.pdf  
- https://arxiv.org/pdf/2107.12146.pdf   
- https://onlinelibrary.wiley.com/doi/epdf/10.1002/pamm.202200306   
## Coherent Point Drift  
https://github.com/neka-nat/probreg   
DL para CPD: https://arxiv.org/pdf/1906.03039.pdf   


## Learning process in pinns  
[[2403.18494] Learning in PINNs: Phase transition, total diffusion, and generalization](https://arxiv.org/abs/2403.18494)

---
# Recursos

### Librerías: 
- Resolución de DiffEq con redes neuronales ([NeuroDiffGym](https://github.com/NeuroDiffGym/neurodiffeq)): 
- Awesome PINNS papers ([[AwesomeLists]]): https://github.com/xgxg1314/My-awesome-PINN-papers 
	- Y el dueño de este repositorio, tiene más repos sobre el tema: https://github.com/xgxg1314 
- [Computer Vision with PINNS]([https://www.linkedin.com/posts/opencv_computervision-quality-stage-activity-7058089339129364480-4GOn?utm_source=share&utm_medium=member_desktop](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/ftp/arxiv/papers/2301/2301.12531.pdf)) ⚠ Ojo: que no es con PINNs, es con info física de optica.  
	- [Repo](https://github.com/JalaliLabUCLA/phycv) 
- Video: [Rethinking Physics Informed Neural Networks](https://www.youtube.com/watch?v=qYmkUXH7TCY) [NeurIPS'21]
- Jax puede que nos sea de utilidad si conseguimos acelerar los cálculos, permitiendo un mejor *real-time*
- 🌟librería para resolver PDEs con PINNs: 
- Nvidia se supone que tiene algo que lo mejora todo: [Modulus - A Neural Network Framework | NVIDIA Developer](https://developer.nvidia.com/modulus)
- Esto parece lo más tocho y avanzado de pinns: [DeepXDE — DeepXDE 1.10.2.dev12+g2236965 documentation](https://deepxde.readthedocs.io/en/latest/)
### Papers
Tengo muchos papers y libros en el Drive de go.ugr, en UPV.  
- [Physics-Guided, Physics-Informed, and Physics-Encoded Neural Networks in Scientific Computing.](https://arxiv.org/pdf/2211.07377.pdf)
- [Monte Carlo PINNs: deep learning approach for forward and inverse problems involving high dimensional fractional partial differential equations](https://arxiv.org/pdf/2203.08501.pdf)
- [Respecting causality is all you need for training physics-informed neural networks](https://arxiv.org/abs/2203.07404)
- [Scientific Machine Learning Through Physics–Informed Neural Networks: Where we are and What’s Next](https://link.springer.com/article/10.1007/s10915-022-01939-z)
- [LIMITATIONS OF PHYSICS INFORMED MACHINE LEARNING FOR NONLINEAR TWO-PHASE TRANSPORT IN POROUS MEDIA](<https://www.dl.begellhouse.com/download/article/415f83b5707fde65/(2)JMLMC-33905.pdf>)
- [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050)
- [The Old and the New: Can Physics-Informed Deep-Learning Replace Traditional Linear Solvers?](https://www.frontiersin.org/articles/10.3389/fdata.2021.669097/full)
- [Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127?via%3Dihub)
- [Adaptive activation functions accelerate convergence in deep and physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999119308411?via%3Dihub)
- Faltan bastante que he metido en la carpeta. 
- [Aquí](https://github.com/AmeyaJagtap/Conservative_PINNs) se citan varios papers interesantes.  
> [!warning] Drawbacks de las pinns
> Translation and discontinuous behavior are hard to approximate using PINNs.[[13]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#cite_note-:3-13) They fail when solving differential equations with slight advective dominance.[[14]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#cite_note-:2-14) They also fail to solve a system of dynamical systems and hence has not been a success in solving chaotic equations.[[25]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#cite_note-25) One of the reasons behind the failure of the regular PINNs is soft-constraining of Dirichlet and Neumann boundary conditions which pose multi-objective optimization problem. This requires the need for manually weighing the loss terms to be able to optimize. Another reason is getting optimization itself. Posing PDE solving as an optimization problems brings in all the problems that are faced in the world of optimization, the major one being getting stuck at a local optimum pretty often.[[14]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#cite_note-:2-14)[[26]](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#cite_note-26)
> [Characterizing possible failure modes in physics-informed neural networks](https://openreview.net/pdf?id=a2Gr9gNFD-J) ,y el [video](https://www.youtube.com/watch?v=qYmkUXH7TCY) en el NeurlIPS
- Book  ->  Physics-Based Deep Learing : [web](https://physicsbaseddeeplearning.org/overview.html) ,   [book](https://arxiv.org/pdf/2109.05237.pdf)  
- Web de los creadores MazziarRaisi: https://maziarraissi.github.io/PINNs/ 
- 🌟[[2308.08468] An Expert's Guide to Training Physics-informed Neural Networks](https://arxiv.org/abs/2308.08468)
### Cursos / recursos
- Ben Moosly introduction: [Physics-Informed Neural Networks (PINNs) - An Introduction - Ben Moseley | The Science Circle - YouTube](https://www.youtube.com/watch?v=G_hIppUWcsc&ab_channel=JousefMuradLITE)
- Curso Brown (Mauricio) -> https://github.com/mvanzulli/IAPCI/tree/main    
  
- Maizi -> https://www.youtube.com/@CrunchGroup    
- En los [seminarios el IPS](https://isp.uv.es/seminars.html) hay algunas charlas sobre ODEs y Physics Based. 
  Por ejemplo: **Discover ODEs from data** -> https://www.youtube.com/watch?v=NmMufP_qDz8  
  - Curso de DeepXDE:[DeepXDE: A Deep Learning Library for Solving Differential Equations by Lu Lu - YouTube](https://www.youtube.com/watch?v=Wfgr1pMA9fY&list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9&index=14&ab_channel=MLPS-CombiningAIandMLwithPhysicsSciences) 
  - Sobre MODULUS DE NVIDIA: [NVIDIA Modulus v22.09 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/index.html)
  - Este curso habla de mucho tema pinns tambien: [Before you continue to YouTube](https://www.youtube.com/playlist?list=PL1e3Jic2_DwwJQ528agJYMEpA0oMaDSA9)