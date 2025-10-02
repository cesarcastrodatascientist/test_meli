import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class ModelPredictor:
    def __init__(self, directorio=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.directorio = Path(directorio) if directorio else base_dir / "data"

    #creando función de lectura del archivo
    def lectura_csv(self):
        ruta = self.directorio / "data_to_model.csv"
        logging.info(f"Leyendo CSV desde: {ruta}")
        return pd.read_csv(ruta)
    
    def train_test_split(self, df, y):
        # Separar variables predictoras (X) y target (y)
        X = df.drop(columns=y)
        y = df[y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test 
    

    def train_model(self, X_train, y_train, model):

        if model=='rf':
        # Definir el modelo base
            model_ = RandomForestRegressor(random_state=42)

            param = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [1, 2, 4]
            }

            grid = GridSearchCV(
                estimator=model_,
                param_grid=param,
                cv=5,             # validación cruzada en 5 folds
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=2
            )

        elif model == 'xgb':
            model_ = XGBRegressor(random_state=42, objective="reg:squarederror")

            param = {
                "n_estimators": [100, 200],
                "max_depth": [5, 7, 10],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.7, 0.8],
                "colsample_bytree": [0.7, 0.8]
            }

            grid = GridSearchCV(
                estimator=model_,
                param_grid=param,
                cv=5,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=2
            )

        return grid.fit(X_train, y_train)
    
    
    def test_model(self, X_test, y_test, model):
        best_model = model
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        return r2, rmse


        
