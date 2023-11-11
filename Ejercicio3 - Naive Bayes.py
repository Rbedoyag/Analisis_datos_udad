
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve



# Leer los datos
ds = pd.read_csv("C:\\Users\\rober\\OneDrive\\Documentos\\UNAD\SEXTO SEMESTRE\\ANÁLISIS DE DATOS\\TAREA4\\Cleaned-Data.csv")
ds.info()


ds.isnull().sum()


# Definir características y objetivos
feature_cols = ['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
                'Sore-Throat', 'None_Sympton', 'Pains', 'Nasal-Congestion',
                'Runny-Nose', 'Diarrhea', 'None_Experiencing']
objetive_col1 = 'Severity_None'
objetive_col2 = 'Severity_Mild'
objetive_col3 = 'Severity_Moderate'
objetive_col4 = 'Severity_Severe'

X = ds[feature_cols]
y1 = ds[objetive_col1]
y2 = ds[objetive_col2]
y3 = ds[objetive_col3]
y4 = ds[objetive_col4]


# Función para evaluar el rendimiento del modelo y visualizar resultados
def evaluate_model(X_train, X_test, y_train, y_test, objetive_col):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Predicciones en el conjunto de prueba
    y_pred = nb.predict(X_test)

    # Métricas de rendimiento
    print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(y_test, y_pred)))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    print('=' * 40)


# División de los datos para cada objetivo
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.3, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.3, random_state=0)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size=0.3, random_state=0)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y4, test_size=0.3, random_state=0)


# Evaluación para cada objetivo
evaluate_model(X_train1, X_test1, y_train1, y_test1, objetive_col1)
evaluate_model(X_train2, X_test2, y_train2, y_test2, objetive_col2)
evaluate_model(X_train3, X_test3, y_train3, y_test3, objetive_col3)
evaluate_model(X_train4, X_test4, y_train4, y_test4, objetive_col4)

