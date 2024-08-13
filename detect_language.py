# TODO: This needs to be adapted
"""
from lingua import Language, LanguageDetectorBuilder

# Usar detect_languages_in_parallel_of para procesar los títulos
# Column text must be provided in df

df.rename(columns={'title': 'text'}, inplace=True)

languages = [Language.ENGLISH, Language.BASQUE, Language.CATALAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()
idiomas_detectados = detector.detect_languages_in_parallel_of(df['text'].tolist())



# Añadir los resultados como una nueva columna en el DataFrame
df['idioma'] = [idioma.name if idioma else 'Indefinido' for idioma in idiomas_detectados]     df_esp = df[df['idioma'] == 'SPANISH’]’’’
"""