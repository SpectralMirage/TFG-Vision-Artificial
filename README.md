# Restauración de postales antiguas

En este fichero se explica la estructura del proyecto

## Carpeta data
    Almacenan los conjuntos de imágenes que se utilizan en la red para los entrenamientos y las predicciones.

## Carpeta images
    Almacenan tres directorios más donde uno contiene las imágenes con las marcas que se quieren eliminar, otro con las imágenes sin marcas y el último con las imágenes generadas a partir de las imágenes de las dos carpetas anteriores

## Carpeta model
    Almacena el modelo obtenido en el último entrenamiento para usarlo durante las predicciones. Al ser un archivo demasiado grande, se ha dividido en siete partes iguales. 
    En el código se utiliza el modelo completo. Para poder juntar las partes del modelo, se puede utilizar el siguiente comando en el directorio del modelo:

    **cat pix2pix_2200_part_* > modelo_reconstruido.pth**     

## Carpeta src
    Contiene todo el código utilizado en el proyecto.

### Pix2Pix notebook
    Este notebook contiene toda la lógica la red Pix2Pix.

### Dataloader
    Script de Python para crear las carpetas de los conjuntos de entrenamiento, validación y test. Para usarlo, es necesario modificar las siguientes variables:

    - folder_path_original y folder_path_gt: Directorios donde se encuentran las imágenes condiciones y ground truth que se van a procesar

    - train_path, val_path y test_path: Directorios donde se van a guardar las imágenes procesadas

### Image threshold
    Script de Python para el aumentado de datos. Utiliza recortes de marcas para insertarlas en imágenes condición y así, generar nuevas imágenes. Para usarlo, es necesario modificar las siguientes variables:

    - path_snippets: Directorio donde se encuentran los recortes de marcas
    - path_images: Directorio de las imágenes condición de las cuales, a partir de ellas, se van a generar nuevas imágenes con las marcas añadidas
    - path_dest: Directorio donde se van a guardar las nuevas imágenes con marcas


