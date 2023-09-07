import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class DataAnalysisPCA:
    def __init__(self, archivo_csv, target_column):
        self.archivo_csv = archivo_csv
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None
        self.X_pca = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None
        self.results = None
    
    def cargar_datos(self):
        self.data = pd.read_csv(self.archivo_csv)
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
    
    def realizar_eda(self):
        # Realiza aquí el análisis exploratorio de datos según tus necesidades
        pass
    
    def aplicar_pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.X_pca = pca.fit_transform(self.X)
    
    def dividir_datos(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_pca, self.y, test_size=test_size, random_state=42)
    
    def entrenar_modelo(self):
        self.clf = RandomForestClassifier(random_state=42)
        self.clf.fit(self.X_train, self.y_train)
    
    def evaluar_modelo(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy
    
    def ejecutar_analisis(self):
        self.cargar_datos()
        self.realizar_eda()
        self.aplicar_pca()
        self.dividir_datos()
        self.entrenar_modelo()
        accuracy = self.evaluar_modelo()
        
        self.results = {
            'PCA Explained Variance Ratio': self.clf.named_steps['pca'].explained_variance_ratio_,
            'Accuracy': accuracy
        }
        
        return self.results

# Ejemplo de uso
if __name__ == "__main__":
    archivo_csv = 'datos.csv'  # Reemplaza con tu archivo CSV
    target_column = 'clase'  # Reemplaza con el nombre de tu columna objetivo
    
    data_analyzer = DataAnalysisPCA(archivo_csv, target_column)
    resultados = data_analyzer.ejecutar_analisis()
    
    # Imprimir los resultados
    print("Resultados:")
    for key, value in resultados.items():
        print(f"{key}: {value}")
