import pandas as pd
import logging
from pathlib import Path
import re
import unicodedata
import numpy as np
from scipy.stats import ttest_ind, f_oneway,pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class FeatureEngineering:
    def __init__(self,directorio=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.directorio = Path(directorio) if directorio else base_dir / "data"

    #creando función de lectura del archivo
    def lectura_csv(self):
        ruta = self.directorio / "data_proccesed.csv"
        logging.info(f"Leyendo CSV desde: {ruta}")
        return pd.read_csv(ruta)

    def eliminar_id_title(self,df):
        df = df.drop(columns=['id','title','date_created'])
        return df
    
    def indicativo_descuento(self,df):
        if (df['base_price']>df['price']).sum()>0:
            df['descuento'] = np.where(df['base_price']>df['price'],1,0)
        else:
            print("Ningún producto presenta descuento")
        return df

    def limpiar_dataframe(self, df, columns):
        
        def limpiar_texto(texto: str) -> str:
            if pd.isna(texto):  # conservar NaN
                return texto
            texto = str(texto).lower()
            texto = unicodedata.normalize("NFD", texto)
            texto = texto.encode("ascii", "ignore").decode("utf-8")
            texto = re.sub(r"\s+", "_", texto)               # espacios → _
            texto = re.sub(r"[^a-z0-9_]", "", texto)         # quitar caracteres especiales
            texto = re.sub(r"_+", "_", texto).strip("_")     # limpiar guiones bajos extras
            return texto
        
        if columns is None:
            columnas = df.select_dtypes(include="object").columns.tolist()
        
        for col in columns:
            df[col] = df[col].apply(limpiar_texto)
        
        return df

    def creando_paretos(self, df, umbral, columns):

        df_copy = df.copy()
        for categoria in columns:
            conteo = df_copy[categoria].value_counts(normalize=False).reset_index()
            conteo.columns = [categoria, "frecuencia"]
            conteo["proporcion"] = conteo["frecuencia"] / conteo["frecuencia"].sum()
            conteo["acumulado"] = conteo["proporcion"].cumsum()
            
            # Categorías que cubren hasta el umbral
            categorias_top = conteo.loc[conteo["acumulado"] <= umbral, categoria].tolist()
            
            if len(categorias_top)>1:
            # Si el conteo de las pareto da mayor a 1 reemplaza en la columna original
                df_copy[categoria] = df_copy[categoria].apply(lambda x: x if x in categorias_top else "otro")
    
        return df_copy

    def analizando_labels(self, df):
        '''La intención es revisar variables categoricas a las cuales al calcular el pareto les queda un solo label
        O les quedan una cantidad inmanejable dentro de un modelo predictivo'''
        columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        lista_drop = []
        for i in columns:
            if df[i].nunique()==1:
                lista_drop.append(i)
            elif df[i].nunique()>=10:
                lista_drop.append(i)
            else:
                print(f"La varible {i} será incluída en el análisis")
        return lista_drop
    
    def test_diferencia_medias(self,df):
        """
            Compara la media de una variable numérica entre grupos definidos por una variable categórica.
            - Si hay 2 grupos -> aplica t-test
            - Si hay más de 2 grupos -> aplica ANOVA
        """
        salida = {}
        drop_columns = []
        for i in df.drop(columns='sold_quantity').select_dtypes(include=["object", "string","bool"]).columns.tolist():
            grupos = [g['sold_quantity'].dropna().values for _, g in df.groupby(i)]
        
            if len(grupos) == 2:  # Comparación entre 2 grupos
                stat, pval = ttest_ind(*grupos)
                test_type = "t-test"
            else:  # Comparación entre más de 2 grupos
                stat, pval = f_oneway(*grupos)
                test_type = "ANOVA"
            if pval > 0.05:
                drop_columns.append(i)

            salida[i] = {
            "test": test_type,
            "estadístico": stat,
            "p_valor": pval,
            "significativo": "Sí" if pval < 0.05 else "No"
            }
        
        return salida,drop_columns
    
    def calcular_correlaciones(self, df):
        """
        Calcula los coeficientes de correlación Pearson y Spearman
        entre dos variables numéricas de un DataFrame.
        """
        salida_cor = {}
        drop_columns = []
        for i in df.drop(columns='sold_quantity').select_dtypes(include=["int","float"]).columns.tolist():
            x, y = df['sold_quantity'], df[i]

            # Pearson
            pearson_corr, pearson_p = pearsonr(x, y)
            pearson_sig = "Sí" if abs(pearson_corr) > 0.75 else "No"
            # Spearman
            spearman_corr, spearman_p = spearmanr(x, y)
            spearman_sig = "Sí" if abs(spearman_corr) > 0.75 else "No"
            if (pearson_sig=='No') & (spearman_sig == 'No'):
                drop_columns.append(i)
            salida_cor[i] = {
                "Pearson": {"coeficiente": pearson_corr, "p_valor": pearson_p,"significativo":pearson_sig },
            "Spearman": {"coeficiente": spearman_corr, "p_valor": spearman_p,"significativo": spearman_sig}
                }
        
        return salida_cor,drop_columns
    
    def bool_to_number(self, df):
        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        for i in bool_cols:
            df[i] = df[i].astype(int)
        return df
    
    def dummies(self, df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype='int')
        return df_dummies
    
    def transform_numericas(self, df):
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def guardar_csv(self,df):
        ruta = self.directorio / "data_to_model.csv"
        logging.info(f"Guardando CSV a la ruta: {ruta}")
        return df.to_csv(ruta,index=False)