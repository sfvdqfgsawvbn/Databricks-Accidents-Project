# Databricks-Accidents-Project

## Présentation du Projet

Ce projet a pour objectif de créer et héberger un modèle prédictif basé sur les données d'accidents de la route. L'ensemble du processus est réalisé sur la plateforme Azure Databricks, avec l'utilisation de différents algorithmes de machine learning tels que KNeighborsClassifier, DecisionTreeClassifier, et RandomForestClassifier.

## Sources des Données

Les données utilisées dans ce projet proviennent des informations sur les accidents de la route. Les détails sur la collecte des données peuvent être trouvés dans le tutoriel d'Ilyes Talbi de la revue IA. Voici comment je mis suis pris pour les données et j'ai fait de même pour chaque csv.

```
# Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris
import pandas as pd

# Charger le jeu de données Iris
iris_data = load_iris()

# Les données sont stockées dans iris_data.data (les caractéristiques) et iris_data.target (les étiquettes/classes)

# Créer un DataFrame pandas pour visualiser les données
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target

# Afficher les 5 premières lignes du DataFrame
print(df.head())
```

## Contenu du Repository

- **Notebooks Databricks**
  - `Modelisation.dbc`: Contient le code pour le prétraitement des données, la construction et l'évaluation des modèles.
  - `Test API.dbc`: Permet de tester le modèle via des exemples d'accidents en utilisant l'API.

- **Azure Configuration**
  - Le code pour la configuration Azure, y compris la création du groupe de ressources, l'instance Databricks, et le cluster.

## Modèle Retenu et Performances

Les modèles construits dans le cadre de ce projet sont évalués en utilisant les métriques d'accuracy et de f1-score. Le meilleur modèle de chaque méthode (KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier) est retenu et enregistré sous le nom "BestModel" dans l'environnement Databricks.

## Endpoint du Modèle

Le modèle retenu est hébergé en tant que point de terminaison sur Databricks. Vous pouvez accéder à l'API en utilisant le lien suivant : 

```
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-8185614889331212.12.azuredatabricks.net/serving-endpoints/1/invocations'
  headers = {'Authorization': f'Bearer {"dapi3498c1d3be79cc8651d19512a8a9a1dd-2"}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()
```

## Utilisation de l'API

Un token a été créé pour permettre la consommation de l'API du modèle. Le délai d'expiration du token est fixé à 30 jours.

## Test du Modèle

Pour tester le modèle, veuillez consulter le notebook `Test API.dbc` dans le dossier Shared du workspace de Databricks.

---

*Note: Toutes les configurations Azure, ainsi que le code et les résultats détaillés sont disponibles dans les notebooks Databricks.*



