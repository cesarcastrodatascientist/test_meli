import pandas as pd
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class DataAnalyzer:
    def __init__(self, directorio=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.directorio = Path(directorio) if directorio else base_dir / "data"

    #creando función de lectura del archivo
    def lectura_csv(self):
        ruta = self.directorio / "new_items_dataset.csv"
        logging.info(f"Leyendo CSV desde: {ruta}")
        return pd.read_csv(ruta)
    
    def resumen_variable(self, data, columns):
        list = ['count','mean','median','std','sum','min','max']
        agg_dict = {col: list for col in columns}
        summary = data.agg(agg_dict).reset_index()
        return summary
    
    def calcular_porcentaje_nas(self,df: pd.DataFrame) -> pd.DataFrame:
        porcentaje_nas = df.isna().mean() * 100
        resultado = (
            porcentaje_nas
            .reset_index()
            .rename(columns={"index": "columna", 0: "porcentaje de na's"})
            .sort_values(by="porcentaje de na's", ascending=False)
            .reset_index(drop=True)
        )
        resultado["numero_na's"] = resultado["porcentaje de na's"]*df.shape[0]/100
        return resultado
    
    def revisar_strings_con_numeros(self, df):
        columnas = df.select_dtypes(include=["object", "string"]).columns.tolist()
        
        reporte = {}
        
        for col in columnas:
        # Detectar registros que son solo números (uno o más dígitos)
            mask = df[col].astype(str).str.match(r"^\d+$", na=False)
            if mask.any():
                reporte[col] = df.loc[mask, col].unique().tolist()
        return reporte
    
    def revisar_variables_tipolista(self, df, columns):
        for i in columns:
            print(f"La variable {i} tiene {round(df[df[i]=='[]'].shape[0]/df.shape[0]*100,2)}% de registros en '[]'")

    def revisar_descuentos(self,df):
        #En teoría el precio base debe ser mayor o igual al precio con descuento
        reporte = {}
        mask = df['base_price']<df['price']
        if mask.any():
            reporte = df[mask].index
        return reporte

    def outliers(self, df, columns,umbral,mostrar_graficos):
        resumen = []

        for col in columns:
            serie = df[col].dropna()
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - umbral * iqr
            upper = q3 + umbral * iqr

            mask = (serie < lower) | (serie > upper)
            outliers = serie[mask]

            resumen.append({
                "columna": col,
                "número_outliers": len(outliers),
                "porcentaje_outliers": f"{round(len(outliers) / len(serie) * 100, 2)}%",
                "lim_inf": lower,
                "lim_sup": upper
            })

            if mostrar_graficos:
                plt.figure(figsize=(6, 3))
                sns.boxplot(x=serie, color="skyblue")
                plt.title(f"Boxplot de {col}")
                plt.show()

        return pd.DataFrame(resumen)

    def guardar_csv(self,df):
        ruta = self.directorio / "data_proccesed.csv"
        logging.info(f"Guardando CSV a la ruta: {ruta}")
        return df.to_csv(ruta,index=False)