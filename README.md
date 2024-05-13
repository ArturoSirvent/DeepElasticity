# Sobre este repositorio:   

Este repositorio contiene los datos y el código para el estudio de la aplicación de PINNs sobre simulaciones de tejido blando, elástico e hiperelástico.  

- Data: Por un lado tenemos los datos simulados por FEM usando como modelo una próstata. Para todos los casos las simulaciones arrojaban desplazamientos y tensiones. Aquí tenemos varias carpeta:
    - En la primera ARTURO_TEST_1 fue una simulación sin suponer la aproximación de pequeños desplazamiento, pero contiene algunos archivos que son reusados constantemente, como NODES.txt y FORCE_ON_NODES.txt, pues no tienen que ver sobre el desplazamiento en sí, sino sobre sobre las condiciones aplicadas sobre el cuerpo, y los ids de los nodos y sus posiciones iniciales.  
    - LINEAR_SMALL_DISP ya supone la aproximación de pequeños desplazamientos, los parámetros del material son los mismo que en los de la simualción de ARTURO_TEST_1.   
    - MULTIPLE_E_VALUES y MULTIPLE_E_VALUES_NEW son más simulaciones con la aprocimación citada, para diferentes valores del parámetro E. Pero con 0.4 de nu.  
    - Para el caso Hiperelástico (002-Neo).  Tenemos una carpeta (ARTURO_TEST_2_NEOH) donde las tensiones están sin las 6 componentes independientes. Y en DATOS_HIPERELASTICO_3 ya tenemos las 6 componenetes.  

- informes: Contiene imágenes usadas para los reports hechos.  
- notebooks: Tiene el desarrollo de código. Por un lado para el caso lineal, y otro para el hiperelástico. 
    - Tenemos los primeros notebooks 000 y 000.1, que son un desarrollo de lo que ya teníamos en el anterior notebook donde estaba realizando el desarrollo (https://github.com/CDiazCuadro/ElasticityPINN rama de linear_prostata, este contenido está en la rama de antiguo desarrollo de este repo).  
    - En 001 probamos la configuración en que el output es desplazamiento y stress.  
    - 002 prueba de las GNNs, resultados que no fueron muy buenos.  
    - 003 sin resultados
    - 004 son las pruebas para la creacion de src.  
    - 005 una agrupacion de resutlados expuestos en el informe hecho para la justificacion de 2023.  
    - 006 limitacion de las posiciones de los collocation points solo dentro de la geometría de la próstata.  
    - Un intento de simulación física son estableciendo la física y las BC, sin data de desplazamientos final. No da buenos resultados.  
- scripts: Tiene el mísmo código que algunos notebooks pero en forma de script para la ejecución desacoplada en la máquina mediante `screen` o `nohup`.  
- src: Las funciones que usamos repetidamente a lo largo de nos notebooks. En algunos notebooks las seguimos definiendo por tener que hacer algún cambio, pero lo presente en src es por lo general la última versión.
    - utils/data.py: Es la clase que agrupa los datos usados en un objeto para usarse para el entrenamiento.  
    - models.py : Los modelos definidos para el entrenamiento. Tenemos la dense network y la PINN también. 
    - train.py : La clase encargada de entrar a la PINN, engloba el bucle de entrenamiento etc.   
