import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import langid
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# all_path = "/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_21_oct_es"
# gencat_path = "/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_31_oct_tr"

# def load_parquet_into_one(input_folder):
#     dfs = []
#     for file_name in tqdm(os.listdir(input_folder)):
#         if file_name.endswith(".parquet"):
#             file_path = os.path.join(input_folder, file_name)
#             df = pd.read_parquet(file_path)
#             dfs.append(df)    
#     return pd.concat(dfs, ignore_index=True) 

# def langDetect(row):
#     try:
#         lang, _ = langid.classify(row.objective.replace("\n", ""))
#     except Exception:
#         lang = "unknown"
#     return lang

# def obtener_id_place(procurement_id_x):
#     url_base = "https://nextprocurement.bsc.es/api/place/"
#     url_fallback = "https://nextprocurement.bsc.es/api/place_menores/"
#     url_completa = f"{url_base}{procurement_id_x}/id"
#     url_completa_fallback = f"{url_fallback}{procurement_id_x}/id"
    
#     try:
#         #print(f"Solicitando ID para {procurement_id_x} con URL principal...")
#         response = requests.get(url_completa)
#         print(url_completa)        
#         if response.status_code == 200:
#             data = response.json()
#             place_id = data.get('id')
#             return place_id
#     except Exception as e:
#         print("Excepción durante la solicitud con la URL principal:", str(e))

#     try:
#         #print(f"Solicitando ID para {procurement_id_x} con URL menores...")
#         response = requests.get(url_completa_fallback)
        
#         if response.status_code == 200:
#             data = response.json()
#             print("El json completo es:", data)  
#             place_id = data.get('id')
#             print("ID obtenido:", place_id)  
#             return place_id
#     except Exception as e:
#         print("Excepción durante la solicitud con la URL secundaria:", str(e))
#     return None

# # Load all data
# df = load_parquet_into_one(all_path)
# print(f"Loaded PLACE data with {len(df)} rows")
# df_gencat = load_parquet_into_one(gencat_path)
# print(f"Loaded TRANSLATED data with {len(df_gencat)} rows")
# df_df_gencat = pd.concat([df, df_gencat])
# print(f"Created final concatenated dataframe with {len(df_df_gencat)} rows")


# # Detect language
# df_df_gencat["lang_objective"] = df_df_gencat.apply(langDetect, axis=1)
# print(f"Detected language for all rows")

# df_df_gencat["procurement_id_f"] = df_df_gencat.apply(    lambda x: x["procurement_id_x"] if not pd.isna(x["procurement_id_x"]) else x["procurement_id"], axis=1)

# df_df_gencat["procurement_id"] = df_df_gencat["procurement_id_f"]

# # drop columns
# df_df_gencat = df_df_gencat.drop(columns=["procurement_id_x", "procurement_id_f"])

# try:
#     df_df_gencat['place_id'] = [obtener_id_place(x) for x in tqdm(df_df_gencat['procurement_id'])]
# except Exception as e:
#     print("Error al obtener los IDs de PLACE:", str(e))
#     import pdb; pdb.set_trace()
    
# df_df_gencat.to_parquet("/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_31_oct_es_translated_with_place_id_created24feb.parquet")


# # insiders / minors
# columns_keep_ins_min = [
#     #'ContractFolderStatus.ContractFolderID', # Expediente
#     'ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode', # CPV
#     'ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode', # Country NUTS
#     'ContractFolderStatus.LocatedContractingParty.ParentLocatedParty.PartyName.Name', # Ubicación orgánica
#     'ContractFolderStatus.ProcurementProject.BudgetAmount.EstimatedOverallContractAmount', # Presupuesto
#     'ContractFolderStatus.TenderResult.SMEAwardedIndicator', # Es pyme
#     'ContractFolderStatus.ContractFolderID', #Expediente
#     'title', # PLACE title (objective)
#     'ContractFolderStatus.ValidNoticeInfo.AdditionalPublicationStatus.AdditionalPublicationDocumentReference.IssueDate', # Date
#     'origin',
#     'place_id'
# ]
# # outsiders
# columns_keep_out = [
#     #'ContractFolderStatus.ContractFolderID', # Expediente
#     'ContractFolderStatus.ProcurementProject.RequiredCommodityClassification.ItemClassificationCode', #CPV
#     'ContractFolderStatus.ProcurementProject.RealizedLocation.CountrySubentityCode', # Country NUTS
#     'ContractFolderStatus.LocatedContractingParty.ParentLocatedParty.PartyName.Name', # Ubicación orgánica
#     'ContractFolderStatus.ProcurementProject.BudgetAmount.EstimatedOverallContractAmount', # Presupuesto
#     'ContractFolderStatus.ContractFolderID', #Expediente
#     'title', # PLACE title (objective)
#     'ContractFolderStatus.ValidNoticeInfo.AdditionalPublicationStatus.AdditionalPublicationDocumentReference.IssueDate', # Date
#     'origin',
#     'place_id'
# ]

# print("Reading from PLACE starts")
# df_PLACE_ins = pd.read_parquet('/export/usuarios_ml4ds/cggamella/sproc/DESCARGA_PLACE_NOV/insiders.parquet')
# df_PLACE_out = pd.read_parquet('/export/usuarios_ml4ds/cggamella/sproc/DESCARGA_PLACE_NOV/outsiders.parquet')
# df_PLACE_min = pd.read_parquet('/export/usuarios_ml4ds/cggamella/sproc/DESCARGA_PLACE_NOV/minors.parquet')

# df_PLACE_out['origin'] = 'out'
# df_PLACE_ins['origin'] = 'ins'
# df_PLACE_min['origin'] = 'min'

# def unify_colname(col):
#     return ".".join([el for el in col if el])

# df_PLACE_ins = df_PLACE_ins.rename(columns={"id":"place_id"})
# df_PLACE_min = df_PLACE_min.rename(columns={"id":"place_id"})
# df_PLACE_out = df_PLACE_out.rename(columns={"id":"place_id"})

# df_PLACE_out.columns = [unify_colname(col) for col in df_PLACE_out.columns]
# df_PLACE_ins.columns = [unify_colname(col) for col in df_PLACE_ins.columns]
# df_PLACE_min.columns = [unify_colname(col) for col in df_PLACE_min.columns]

# df_PLACE_out = df_PLACE_out[columns_keep_out]
# df_PLACE_out = df_PLACE_out.loc[:, ~df_PLACE_out.columns.duplicated()]

# df_PLACE_ins = df_PLACE_ins[columns_keep_ins_min]
# df_PLACE_ins = df_PLACE_ins.loc[:, ~df_PLACE_ins.columns.duplicated()]

# df_PLACE_min = df_PLACE_min[columns_keep_ins_min]
# df_PLACE_min = df_PLACE_min.loc[:, ~df_PLACE_min.columns.duplicated()]

# df_PLACE_all = pd.concat([df_PLACE_min, df_PLACE_ins, df_PLACE_out], ignore_index=True)

# print("PLACE created")

# try:
#     df_df_gencat = pd.merge(df_df_gencat, df_PLACE_all, on="place_id", how="left")
# except Exception as e:
#     print(e)
#     import pdb; pdb.set_trace()
    
# df_df_gencat.to_parquet("/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_31_oct_es_translated_with_place_id_created24feb_merge_with_place.parquet")

path_save = "/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_31_oct_es_translated_with_place_id_created24feb_merge_with_place.parquet"
df_df_gencat = pd.read_parquet("/export/data_ml4ds/NextProcurement/PLACE/pliegos_objectives/all_extracted_31_oct_es_translated_with_place_id_created24feb_merge_with_place.parquet")


model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def compute_similarity(row, col1, col2):
    text1, text2 = row[col1], row[col2]
    
    if pd.notna(text1) and pd.notna(text2):  # Ensure both are not None
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return similarity
    return np.nan  # If any value is None, return NaN

def add_similarity_column(df, col1, col2, new_col="similarity"):
    df[new_col] = df.apply(lambda row: compute_similarity(row, col1, col2), axis=1)
    return df


df_df_gencat = add_similarity_column(df_df_gencat, "objective", "title")

df_df_gencat.to_parquet(path_save)