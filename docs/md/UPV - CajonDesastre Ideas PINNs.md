# UPV - CajonDesastre Ideas PINNs  
2023-05-23 (Tuesday) | 16:43   
Type: #notebook  
Main: [[Main - UPV]]
Tags: #DeepLearning , #idea  
Relates: [[ToDo Miscelanea]], [[Topic - PINNS]], [[UPV - PINNS]]  
Status: 

---
# About

Todo lo que sobra y no se donde ponerlo para que no se pierda sobre ideas etc de PINNS y lo de la UPV.   



# Ideas  
- [[DL - GraphNeuralNetworks]]
- Uso de jax y cosas desarrolladas entorno a esto: 
  ❗PUEDE QUE SEA MUY RELEVANTE SI QUEREMOS **OPTIMIZAR Y ACELERAR**
	- [equinox](https://github.com/patrick-kidger/equinox/tree/main):elegant NNs
	- [Diffrax](https://github.com/patrick-kidger/diffrax): numerical ODE/SDE solver.  
- 🌟🌟Temas sobre Neural Differential Equations -> https://arxiv.org/pdf/2202.02435.pdf 
- Transformers con información física ¿?  [[DL - PINNs + Transformers]]
- Las **DeepONets** parece algo tochísimo en la linea de las PINNs.    
- Self-adaptative physics informed neural nets. Esto estaba entre lo que me paso MJ al principio de un curso de noseque:   
- ![[Captura de pantalla 2022-10-27 a las 17.27.57.png|600]]
  En esta imagen vemos temas de self-adaptative PINNS donde la importancia de cada valor para la pinns se ajusta, esto podría usarse para descubrir parámetros o ecuaciones constitutivas.  
- 
> [!NOTE]- Ideas trabajo TFM Ruben PINNS
> En el trabajo de Ruben habla de las funciones de activación. Esto es muy importante para que las derivadas existan, y la solución esté bien construida en el espacio de las funciones de activación. Si nuestra solución es periódica, alomejor sería conveniente algo periodico como función de activación, no se.   
>   ![[Pasted image 20230528122107.png]]
>   También pone limitaciones sobre la función de pérdida, pues tiene que cumplir: 1 con los datos, 2 con la física, 3 con restricciones de la física que no están explicitamente en las ecuaciones, por ejemplo, la solución buscada en el caso de las orbitas del problema de ruben, no podía tener excentricidades negativas etc...   
> ![[Pasted image 20230528122355.png]]     
>   Esto se puede mejor en contexto en el trabajo.     

  - Topological features ([TDA-Net](https://arxiv.org/pdf/2101.08398.pdf)):  En este paper hablar de converting a given tensor (i.e.,a multidimensional array such as a 2d image), to a vectorized representation of its persistence diagram. Como funciona esto? Parece muy interesante.  -> [[TDA-net.pdf]]   
    Todo esto va en la linea del 
- Las PINNs reducen el espacio de parametros, al parecer hay otras formas de tambien hacer eso, noseque de los pesos en la esfera unidad: https://arxiv.org/pdf/1805.08340.pdf  

- Añadir el Gradient Checking de alguna manera?   
  ![[Pasted image 20231030220425.png|300]]  
- Imagen chula de la loss: [(a) Schematic of a PINN for solving inverse problem in photonics based... | Download Scientific Diagram](https://www.researchgate.net/figure/a-Schematic-of-a-PINN-for-solving-inverse-problem-in-photonics-based-on-partial_fig1_340168418) 