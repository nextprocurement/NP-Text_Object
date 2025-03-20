import pathlib
import logging
import re
import unicodedata

OBJETOS = [
    'OBJETO DEL PLIEGO DE PRESCRIPCIONES TÉCNICAS PARTICULARES El presente Pliego de',
    'OBJETO. El presente pliego establece',
    'OBJETO El presente pliego establece',
    'Objeto: El presente pliego establece',
    'OBJETO El presente proyecto de Obras',
    'OBJETO El objeto del contrato es',
    'OBJETO El objeto de este Pliego es',
    'OBJETO DEL CONTRATO El presente contrato tiene por objeto',
    'OBJETO DEL CONTRATO. El presente documento tiene por objeto',
    'OBJETO DEL CONTRATO El presente documento tiene por objeto',
    'OBJETO DEL CONTRATO El objeto del presente contrato',
    'OBJETO DEL PRESENTE CONTRATO En el presente documento',
    'OBJETO DEL SERVICIO Es el objeto de este procedimiento',
    'OBJETO El presente Pliego de Prescripciones Técnicas Particulares (PPT) tiene por objeto',
    'Objeto del pliego El presente documento tiene por objeto',
    'Objeto.– El presente pliego tiene por objeto',
    'OBJETO DEL PROYECTO El presente proyecto de'
    'El objeto de este contrato es',
    'El objetivo del contrato es',
    'El objeto del contrato es',
    'El objeto del presente pliego de prescripciones técnicas es',
    'El presente Pliego de Prescripciones Técnicas tiene por objeto',
    'El presente Pliego de Condiciones Técnicas tiene por objeto',
    'El objeto de las presentes condiciones técnicas particulares',
    'El presente Pliego de Condiciones particulares del Proyecto tiene por finalidad',
    'El presente pliego tiene por objeto establecer',
    'OBJETO del PLIEGO El objeto de las presentes condicionesTécnicas particulares',
    r'Objetivos del contrato de servicios y duración(?!\s+\w+\s+\d+\b)',
    'Este Pliego de Prescripciones Técnicas tiene por objeto establecer',
    'Este Proyecto tiene por objeto',
    'La presente propuesta tiene por objeto',
    'El objetivo de este pliego de condiciones es',
    'Obra Objeto del contrato',
    'Obra Objeto del contrato:',
    'OBJETO DEL PROYECTO El objeto del presente proyecto es',
    'El objeto del presente Proyecto',
    'El objeto del presente pliego',
    'El objeto de este pliego es',
    'El objeto del Presente contrato',
    'El objetivo de este proyecto',
    'Objeto del encargo',
    '3 OBJETO DEL PROYECTO 3.1 Objetivos',
    'OBJETO DEL PROYECTO 3 1 Objetivos',
    'El objeto de la presente convocatoria',
    'OBJETO. 1.1.–',
    'OBJETO 1 1',
    'Objeto del encargo',
    'OBJETO DEL PRESENTE CONTRATO',
    'OBJETO DEL PLIEGO',
    'OBJETO DE LA CONTRATACIÓN',
    'OBJETO DE LA CONTRATACION',
    'OBJETO DEI CONTRATO',
    'OBJETO DEL CONTRATO',
    'OBJETIVO DEL CONTRATO',
    'OBJETO DEL PROYECTO',
    'OBJETO DEL SERVICIO',
    'OBJETO DE ESTE PROYECTO',
    'OBJECTE DE LA CONTRACTACIÓ',
    'OBJETIVOS DEL CONTRATO',
    'OBJETO DEL PROCEDIMIENTO DE CONTRATACIÓN',
    'INFORMACIÓN SOBRE EL PROCEDIMIENTO DE CONTRATACIÓN',
    'OBRAS INCLUIDAS EN EL PLIEGO',
    'OBJETO Y LOTES',
    'OBJETO'
]

def init_logger(name: str, path_logs: pathlib.Path) -> logging.Logger:
    """
    Initialize a logger with a specified name and log file path.

    Parameters
    ----------
    name : str
        The name of the logger.
    path_logs : Path
        The directory path where the log file will be stored.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create path_logs dir if it does not exist
    path_logs.mkdir(parents=True, exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(path_logs / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_format)
    console_handler.setFormatter(console_format)

    # Add the handlers to the logger if they are not already added
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def find_object_pos(texto, objetos):
    texto = texto.lower().strip()
    texto_corto = texto[:1000000]

    mejores_resultados = []

    for obj in objetos:
        obj_lower = obj.lower().strip()

        # No escapamos el objeto si ya es un patrón regex
        if "(" in obj_lower or "[" in obj_lower or "?" in obj_lower:
            regex_obj = obj_lower
        else:
            regex_obj = re.escape(obj_lower)
            #regex_obj = r"\b" + re.escape(obj_lower) + r"\b(?!\.)"

        matches = list(re.finditer(regex_obj, texto_corto))
        #print(matches)

        # Guardamos todas las coincidencias en la lista
        for match in matches:
            mejores_resultados.append((obj, match.start(), len(match.group())))

    if mejores_resultados:
        # Crear un diccionario con las prioridades según el índice en objetos
        prioridad_objetos = {obj.lower().strip(): i for i, obj in enumerate(objetos)}
    
        # Ordenar por prioridad en la lista de objetos y luego por posición en el texto
        mejores_resultados.sort(key=lambda x: (prioridad_objetos.get(x[0].lower().strip(), float('inf')), x[1]))
    
        return mejores_resultados[0][0], mejores_resultados[0][1]


    return None, -1  