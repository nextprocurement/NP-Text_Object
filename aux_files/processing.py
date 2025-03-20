import os
import pandas as pd
from tqdm import tqdm 
tqdm.pandas()  # This enables progress tracking for pandas
#from sentence_transformers import SentenceTransformer
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
pd.set_option('display.max_colwidth', None)

base_url = "https://nextprocurement.bsc.es/api/place/XXX/"
url_menors = "https://nextprocurement.bsc.es/api/place_menores/XXX/"

path = "/export/data_ml4ds/NextProcurement/PLACE/Septiembre2024"
output_path = "/export/data_ml4ds/NextProcurement/PLACE/Diciembre_2024_procesados_all"

################################################################################
def get_admin_tech(procurement_id):

    urls = [base_url, url_menors]
    admin = "Datos_Generales_del_Expediente/Pliego_de_Clausulas_Administrativas/Archivo"
    tecnicas = "Datos_Generales_del_Expediente/Pliego_de_Prescripciones_Tecnicas/Archivo"
    
    for url in urls:
        try:
            formatted_url = url.replace("XXX", procurement_id)
            #print(formatted_url)
            response = requests.get(formatted_url)
            response.raise_for_status()

            data = response.json()
            
            admin = data.get(admin, None)
            tecnicas = data.get(tecnicas, None)
            
            return admin, tecnicas
        except requests.RequestException as e:
            print(f"Request error for {procurement_id}: {e}")
            return None
        except Exception as e:  # Catch other potential exceptions
            print(f"General error for {procurement_id}: {e}")
            return None
        
        except requests.RequestException:
            continue
    return None


def is_minor(x):
    formatted_url = base_url.replace("XXX", x)
    response = requests.get(formatted_url)
    return True if ("httpCode" in response.json() and response.json()["httpCode"] == 404) else False

################################################################################

"""
print(f"-- -- Reading from BSC starts")
all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parq')]
df_all = []
for file in tqdm(all_files, desc="Procesando archivos"):
    df = pd.read_parquet(file)
    df_all.append(df)
df = pd.concat(df_all)
#df.to_csv("all_files.csv", index=False)

print("-- -- Processed files from BSC")

doc_names_by_procurement = df.groupby('procurement_id')['doc_name'].unique()

result = doc_names_by_procurement.reset_index()
result.to_csv("doc_names_by_procurement.csv", index=False)
print(f"Extracted unique procurement IDS")
print(result)


result["admin"] = result["doc_name"].apply(lambda x: next((item for item in x if "admin" in item), None))
result["tech"] = result["doc_name"].apply(lambda x: next((item for item in x if "tecnica" in item), None))

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(is_minor, x): x for x in procurement_ids}
    for future in as_completed(futures):
        results.append(future.result())
        count += 1
        
        # Log progress every 100 items
        if count % 100 == 0:
            print(f"Processed {count} out of {total} items")

# Save results to a CSV file
result["is_minor"] = results
result.to_csv("processed_results.csv", index=False)

# Output the result
print("Processing complete. Results saved to 'processed_results.csv'.")
print(result)
"""

result = pd.read_csv("processed_results.csv")
procurement_ids = result["procurement_id"].values.tolist()

print(f"-- -- Processing {len(procurement_ids)} items")
results_tech = []
results_admin = []
count = 0
total = len(procurement_ids)  # Total number of items for reference

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_admin_tech, x): x for x in procurement_ids}
    for future in as_completed(futures):
        future_result = future.result()
        if result is not None:
            admin, tech = future_result
            results_admin.append(admin)
            results_tech.append(tech)
        else:
            results_admin.append(None)  # Append None to maintain consistency
            results_tech.append(None) 
        count += 1
        
        # Log progress every 100 items
        if count % 100 == 0:
            print(f"Processed {count} out of {total} items")

import pdb; pdb.set_trace()
# Save results to a CSV file
result["admin_from_api"] = results_admin
result["tech_from_api"] = results_tech
result.to_csv("processed_results_with_admin_tech_from_source.csv", index=False)

# Output the result
print(result)

