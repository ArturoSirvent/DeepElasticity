# UPV - Mecánica de los medios continuos  
2023-05-27 (Saturday) | 21:20   
Type: #notebook  
Main: [[Topic - Physics]]
Tags: #fisica  #UPV   
Relates:  [[UPV - PINNS]]
Status: 

---
# About
- Book: Elasticdad Canelas FING


---
# Contenido   



**Sobre los esfuerzos internos:**  
![[Pasted image 20230528131914.png]]
También da una buena definición de la M. de los M. Continuos

![[Pasted image 20230529164819.png]]

---
# Centrado en FEM  
No es es lo mismo estudiar elasticidad, que elasticidad enfocada a los elementos finitos.   
La formulación usual de un problema de elasticidad esta basado en métodos variacionales.  
La energía potencial del sistema es la energía almacenada en la estructura deformada, menos la energía de las fuerzas que actúan sobre este. Es decir, imagina que un elemento elástico tiene un peso encima, tenemos una energía potencial almacenada en el sistema positiva, porque esta deformado y quiere volver a su posición, pero el peso que tiene encima, ha realizado esa misma cantidad de trabajo para deformarlo. Entonces yo diría que la energía potencial del sitema es 0, pero capaz que no.     
El principio de usar los [[Metodos variacionales]] es que la energía $\Pi_P$ será mínima en el equilibrio, dando lugar a un particular campo de desplazamientos $u$ que satisfará las ecuaciones diferenciales del sistema y las condiciones de contorno. Si ese punto encontrado es mínimo, entonces encontraremos que: $$\frac{\partial \Pi_p (u_i)}{\partial u_i  }=0$$ La deformación de posiciones nos viene data por un tensor de gradientes de deformación, el cual nos indica como varían las posiciones.  
Tendremos 3 direcciones principales de deformación, y podemos expresar ese tensor de gradientes de deformación en función de estas.  $$F=\sum_{\alpha=1}^3 \lambda_\alpha n_\alpha \otimes  N_\alpha$$
Con las lambda las elongaciones en las 3 direcciones principales: $\lambda = 1+ \frac{dL}{L}$, N son los vectores material, y n son los vectores espaciales.  

Por un lado tenemos el vector *Chauchy Green deformation tensor* -> **B** que nos indicará las deformaciones, y por otro tenemos el tensor de esfuerzos/tensiones/estress, Cauchy -> **T**
, que nos describirá las tensiones. Y las relaciones son las siguientes:     
![[Pasted image 20230529175853.png]]  

> [!NOTE] 
> Under the assumption of isotropic behavior, the strain
> energy density function Ws can be expressed as a function of the
> strain invariants (Rivlin, 1948a; 1948b):
> Ws = Ws (I1 , I2 , I3 )
> Alternatively, Ws can also be expressed directly as a function of
> the three principal stretches (Valanis & Landel, 1967), namely λ1 ,
> λ2 and λ3 .





## Elasticidad Lineal  
![[Pasted image 20230601192306.png]]  

Esto siguiendo un libro que es Introducción a la elasticidad lineal, que es la ostia.  


Esto lo sigo en el articulo de elasticidad [[Physics - Elasticidad y hiperelasticidad]]


## Gradiente de deformación:  
https://www.continuummechanics.org/deformationgradient.html  


---
# Recursos  

## Libros  


### Mecánica de los medios continuos     
6. L.I. Sedov, A course in Continuum Mechanics, Ed. Walter/Noordhoff, 1971.    
7. H. Heinbockel, Introduction to Tensor Calculus and Continuum Mechanics, Department of Mathematics and Statistics, Old Dominion University, 1996   
8. E. Levy, Elementos de mecánica del medio continuo, Ed. Limusa-Wiley, 1971. 
9. S.C. Hunter, Mechanics of Continuous Media, Ed. Ellis Horwood/John Wiley, 1983.    
10. T.J. Chung, Continuum Mechanics, Rd. Prentice-Hall Inc., 1988.   
11. I.S. Sokolnikoff, Análisis tensorial, Index-Prial, 1971.    
12. I.S. Sokolnikoff, Mathematical Theory of Elasticity, McGraw Hill, 1956.  

## Web  
Esta página está super bien porque pone ejemplo bastante concretos: -> https://www.continuummechanics.org/