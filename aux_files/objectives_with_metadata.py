import os
import pandas as pd
import argparse
import requests
import numpy as np
from tqdm import tqdm
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def obtener_id_place(procurement_id_x):
    url_base = "https://nextprocurement.bsc.es/api/place/"
    url_fallback = "https://nextprocurement.bsc.es/api/place_menores/"
    url_completa = f"{url_base}{procurement_id_x}/id"
    url_completa_fallback = f"{url_fallback}{procurement_id_x}/id"
    
    try:
        print(f"Solicitando ID para {procurement_id_x} con URL principal...")
        response = requests.get(url_completa)
        
        if response.status_code == 200:
            data = response.json()
            print("El json completo es:", data)  
            place_id = data.get('id')
            print("ID obtenido:", place_id)  
            return place_id
        else:
            print("Respuesta no exitosa con la URL principal. Código de estado:", response.status_code)
    
    except Exception as e:
        print("Excepción durante la solicitud con la URL principal:", str(e))

    try:
        print(f"Solicitando ID para {procurement_id_x} con URL menores...")
        response = requests.get(url_completa_fallback)
        
        if response.status_code == 200:
            data = response.json()
            print("El json completo es:", data)  
            place_id = data.get('id')
            print("ID obtenido:", place_id)  
            return place_id
        else:
            print("Respuesta no exitosa con la URL menores. Código de estado:", response.status_code)
    
    except Exception as e:
        print("Excepción durante la solicitud con la URL secundaria:", str(e))

    print("No se pudo obtener el ID ni con la URL principal ni con la de menores.")
    return None

def asignar_ccaa_provincia(df_cruce, mapeo_nuts_ccaa, mapeo_nuts_provincia):
    def obtener_ccaa(code):
        if pd.isna(code) or code in ['', 'None']:
            return np.nan
        try:
            xy = int(code[2:4])
            return mapeo_nuts_ccaa.get(xy, np.nan)
        except:
            return np.nan

    def obtener_provincia(code):
        if pd.isna(code) or code in ['', 'None']:
            return np.nan
        try:
            xyz = int(code[2:5])
            return mapeo_nuts_provincia.get(xyz, np.nan)
        except:
            return np.nan

    df_cruce['ccaa'] = df_cruce['ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode'].apply(obtener_ccaa)
    df_cruce['provincia'] = df_cruce['ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode'].apply(obtener_provincia)
    return df_cruce

# Definición de las columnas a mantener
columns_keep_ins_min = [
    'ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode', #CPV
    'ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode', # Ciudad NUTS
    'ContractFolderStatus.LocatedContractingParty.ParentLocatedParty.PartyName.Name', #Ubicación orgánica
    'ContractFolderStatus.ProcurementProject.BudgetAmount.EstimatedOverallContractAmount', #Presupuesto
    'ContractFolderStatus.TenderResult.SMEAwardedIndicator', #Es pyme
    'ContractFolderStatus.ContractFolderID', #Expediente
    'title', #Título de PLACE
    'ContractFolderStatus.ValidNoticeInfo.AdditionalPublicationStatus.AdditionalPublicationDocumentReference.IssueDate', #Fecha
    'origin',
    'place_id'
]
# En outsiders no se determina si es PYME o no.
columns_keep_out = [
    'ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode', #CPV
    'ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode', # Ciudad NUTS
    'ContractFolderStatus.LocatedContractingParty.ParentLocatedParty.PartyName.Name', #Ubicación orgánica
    'ContractFolderStatus.ProcurementProject.BudgetAmount.EstimatedOverallContractAmount', #Presupuesto
    'ContractFolderStatus.ContractFolderID', #Expediente
    'title', #Título de PLACE
    'ContractFolderStatus.ValidNoticeInfo.AdditionalPublicationStatus.AdditionalPublicationDocumentReference.IssueDate', #Fecha
    'origin',
    'place_id'
]

mapeo_nuts_ccaa = {
    11: "Galicia",
    12: "Principado de Asturias",
    13: "Cantabria",
    21: "Pais Vasco",
    22: "Comunidad Foral de Navarra",
    23: "La Rioja",
    24: "Aragon",
    30: "Comunidad de Madrid",
    41: "Castilla y Leon",
    42: "Castilla-La Mancha",
    43: "Extremadura",
    51: "Cataluña",
    52: "Comunitat Valenciana",
    53: "Illes Balears",
    61: "Andalucia",
    62: "Region de Murcia",
    63: "Ciudad de Ceuta",
    64: "Ciudad de Melilla",
    70: "Canarias"
}

mapeo_nuts_provincia = {
    111: "A Coruña",
    112: "Lugo",
    113: "Ourense",
    114: "Pontevedra",
    120: "Asturias",
    130: "Cantabria",
    211: "Araba/Álava",
    212: "Gipuzkoa",
    213: "Bizkaia",
    220: "Navarra",
    230: "La Rioja",
    241: "Huesca",
    242: "Teruel",
    243: "Zaragoza",
    300: "Madrid",
    411: "Avila",
    412: "Burgos",
    413: "León",
    414: "Palencia",
    415: "Salamanca",
    416: "Segovia",
    417: "Soria",
    418: "Valladolid",
    419: "Zamora",
    421: "Albacete",
    422: "Ciudad Real",
    423: "Cuenca",
    424: "Guadalajara",
    425: "Toledo",
    431: "Badajoz",
    432: "Cáceres",
    511: "Barcelona",
    512: "Girona",
    513: "Lleida",
    514: "Tarragona",
    521: "Alicante",
    522: "Castellón",
    523: "Valencia",
    531: "Ibiza y Formentera",
    532: "Mallorca",
    533: "Menorca",
    611: "Almería",
    612: "Cádiz",
    613: "Córdoba",
    614: "Granada",
    615: "Huelva",
    616: "Jaén",
    617: "Málaga",
    618: "Sevilla",
    620: "Murcia",
    630: "Ceuta",
    640: "Melilla",
    704: "Fuerteventura",
    705: "Gran Canaria",
    708: "Lanzarote",
    703: "El Hierro",
    706: "La Gomera",
    707: "La Palma",
    709: "Tenerife"
}

def process_parquets(df_extracted, df_PLACE, output_path):
    # Obtener place_id
    df_extracted['place_id'] = [obtener_id_place(x) for x in tqdm(df_extracted['procurement_id_x'])]
    df_cruce = pd.merge(df_extracted, df_PLACE, on="place_id", how="inner")
    df_cruce = asignar_ccaa_provincia(df_cruce, mapeo_nuts_ccaa, mapeo_nuts_provincia)
    # Guardar el resultado
    df_cruce.to_parquet(output_path)

def main():
    
    # All PLACE data
    path_PLACE = '/export/usuarios_ml4ds/cggamella/NP-Text_Object/data/objectives_with_metadata/df_PLACE_completo.parquet'
    df_PLACE = pd.read_parquet(path_PLACE)
    logger.info("Archivo con todas las licitaciones de PLACE cargado")
    logger.info("*" * 50)
    
    parser = argparse.ArgumentParser(description="Procesar archivos parquet individuales o por carpeta.")
    parser.add_argument("--input_file", type=str, help="Ruta al archivo parquet individual con objetivos contrato extraídos")
    parser.add_argument("--input_folder", type=str, help="Ruta a la carpeta con archivos parquet")
    parser.add_argument("--output_path", type=str, required=True, help="Ruta de salida para los archivos procesados (directorio si es carpeta)")

    args = parser.parse_args()

    if args.input_file:
        logger.info("Ejecutando en modo un archivo...")
        df_extracted = pd.read_parquet(args.input_file)
        output_file = os.path.join(args.output_path, os.path.basename(args.input_file).replace('.parquet', '_processed.parquet'))
        process_parquets(df_extracted, df_PLACE, output_file)

    elif args.input_folder:
        logger.info("Ejecutando en modo una carpeta entera...")
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        for parquet_file in os.listdir(args.input_folder):
            if parquet_file.endswith(".parquet"):
                input_file_path = os.path.join(args.input_folder, parquet_file)
                df_extracted = pd.read_parquet(input_file_path)
                
                output_file = os.path.join(args.output_path, parquet_file.replace('.parquet', '_processed.parquet'))
                process_parquets(df_extracted, df_PLACE, output_file)
    else:
        print("Debes especificar --input_file(.parquet) o --input_folder(containing .parquet).")

if __name__ == "__main__":
    main()
    
    
## Ejecutar:
# TODA LA CARPETA
#python3 objectives_with_metadata.py --input_folder /export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives_clean --output_path /export/usuarios_ml4ds/cggamella/NP-Text_Object/data/objectives_with_metadata/procesado/
# UN ARCHIVO
#python3 objectives_with_metadata.py --input_file /export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_12aug_es/all_extracted_12aug_es_4_objectives.parquet --output_path /export/usuarios_ml4ds/cggamella/NP-Text_Object/data/objectives_with_metadata/procesado/