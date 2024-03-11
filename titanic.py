import pandas as pd
import numpy as np

data_raw = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQoP0fj0IM32zsvXZz-qzQfCk1f8qpSVGhs3SuJkq50EL0iVCT2_R6D5WxoqpnWmi-PcNlETAj8dK2S/pub?output=csv')
data_test  = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1UrNSppjTMNRQCsqXtfNfWuIunUl0woBPBuOwlJnna7wN8r5nHcDbHmeBcq6P6jlUGmOTr4eZUwhm/pub?output=csv')
data_gender_submission = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRbl_FdeObBBE5a3d6ed0yRcK9sqmsSjAu8SOsZz3mFCRUFWW_G2ZtQZLCXD0TkZgPb8ZZHSLXXcces/pub?output=csv')

# Eliminar la variable "PassengerId"
data_raw = data_raw.drop(columns=['PassengerId'])

# Función con patrón de expresión regular para extraer el título del nombre
def extract_name(df, var):
    pattern = r',\s*([^\.]+)\.'
    title_name = df[var].str.extract(pattern, expand=False)
    return title_name

data_raw['title'] = extract_name(data_raw, 'Name')
data_test['title'] = extract_name(data_raw, 'Name')

# Eliminar variable "Name"
data_raw = data_raw.drop(columns=['Name'])
data_test = data_test.drop(columns=['Name'])

# Hay algunos títulos que se podrían agrupar
data_raw['title'] = data_raw['title'].replace(['Mlle', 'Ms'], 'Miss')
data_raw['title'] = data_raw['title'].replace('Mme', 'Mrs')
data_raw['title'] = data_raw['title'].replace(['Capt', 'Col', 'Major'], 'Military')
data_raw['title'] = data_raw['title'].replace(['Don', 'Sir', 'Jonkheer'], 'Noble')
data_raw['title'] = data_raw['title'].replace(['Lady', 'the Countess'], 'Lady')

data_test['title'] = data_test['title'].replace(['Ms'], 'Miss')
data_test['title'] = data_test['title'].replace(['Col'], 'Military')
data_test['title'] = data_test['title'].replace(['Dona'], 'Lady')

# Revisar nuevamente por títulos diferentes
count_titles_merged_raw = data_raw['title'].value_counts()
count_titles_merged_test = data_test['title'].value_counts()

# Mapear los títulos de los pasajeros
map_title = {
    'Mr': 1,
    'Miss': 2,
    'Mrs': 3,
    'Master': 4,
    'Dr': 5,
    'Rev': 6,
    'Military': 7,
    'Noble': 8,
    'Lady': 9
}

data_raw['title_id'] = data_raw['title'].map(map_title)
data_test['title_id'] = data_test['title'].map(map_title)

# Eliminar la variable "title"
data_raw = data_raw.drop(columns=['title'])
data_test = data_test.drop(columns=['title'])

# Mapear los sexos de los pasajeros
map_sex = {
    'male': 1,
    'female': 2,
}

data_raw['sex_id'] = data_raw['Sex'].map(map_sex)
data_test['sex_id'] = data_test['Sex'].map(map_sex)

# Eliminar la variable "Sex"
data_raw = data_raw.drop(columns=['Sex'])
data_test = data_test.drop(columns=['Sex'])

# Para la variable "Age" se modifican sus valores NULL por el promedio
def mean(df, var):
    mean = df[var].mean()
    return mean

data_raw['Age'] = data_raw['Age'].fillna(mean(data_raw, 'Age'))
data_test['Age'] = data_test['Age'].fillna(mean(data_test, 'Age'))

# Redondear "Age" a 2 decimales
data_raw['Age'] = data_raw['Age'].round(2)
data_test['Age'] = data_test['Age'].round(2)

# Función para unificar las variables "SibSp" y "Parch"
def var_sum (df, var1, var2):
    sum = df[var1] + df[var2]
    return sum

family_size_raw = var_sum(data_raw, 'SibSp', 'Parch')
family_size_test = var_sum(data_test, 'SibSp', 'Parch')

data_raw['family_size'] = family_size_raw
data_test['family_size'] = family_size_test

# Función para crear nueva variable que identifique si el pasajero viaja solo
map_family_size = lambda x: 0 if x == 0 else 1

data_raw['is_alone'] = data_raw['family_size'].map(map_family_size)
data_test['is_alone'] = data_test['family_size'].map(map_family_size)

# Eliminar las variables de "SibSp" y "Parch"
data_raw = data_raw.drop(columns=['SibSp'])
data_test = data_test.drop(columns=['SibSp'])

data_raw = data_raw.drop(columns=['Parch'])
data_test = data_test.drop(columns=['Parch'])

# Eliminar la variable "Ticket"
data_raw = data_raw.drop(columns=['Ticket'])
data_test = data_test.drop(columns=['Ticket'])

# Eliminar la variable "Cabin"
data_raw = data_raw.drop(columns=['Cabin'])
data_test = data_test.drop(columns=['Cabin'])

# Mapear la variable "Embarked" 
map_embarked = {
    'C': 1,
    'Q': 2,
    'S': 3,
}

data_raw['embarked_id'] = data_raw['Embarked'].map(map_embarked)
data_test['embarked_id'] = data_test['Embarked'].map(map_embarked)

# Eliminar la variable "Embarked"
data_raw = data_raw.drop(columns=['Embarked'])
data_test = data_test.drop(columns=['Embarked'])

# Función para buscar la mediana y reemplazar los valores NULL
def median(df, var):
    median = df[var].median()
    return median

data_raw['embarked_id'] = data_raw['embarked_id'].fillna(median(data_raw, 'embarked_id'))
data_test['embarked_id'] = data_test['embarked_id'].fillna(median(data_test, 'embarked_id'))

# se buca la mediana y se reemplazan los valores NULL
data_test['Fare'] = data_test['Fare'].fillna(median(data_test, 'Fare'))

# Redondear "Fare" a 2 decimales
data_raw['Fare'] = data_raw['Fare'].round(2)
data_test['Fare'] = data_test['Fare'].round(2)

# Con respecto a "title_id", al ser varias clases no es posible utilizar el promedio, 
# por lo cual también se utilizará la mediana

data_test['title_id'] = data_test['title_id'].fillna(median(data_test, 'title_id'))

# Función para cambiar las mayúsculas a minúsculas y seguir camel case.
def convertir_de_minusculas_a_mayusculas(texto):
    nuevo_texto = ''
    for letra in texto:
        if letra.isupper():
            nuevo_texto += letra.lower()
        else:
            nuevo_texto += letra
    return nuevo_texto

data_cleaned = data_raw.rename(columns=lambda x: convertir_de_minusculas_a_mayusculas(x))
data_test = data_test.rename(columns=lambda x: convertir_de_minusculas_a_mayusculas(x))

# Finalmente, se mueve la variable a predecir (survived) al final.
columna_a_mover = data_cleaned.pop('survived')

# Insertar la columna al final del DataFrame
data_cleaned.insert(len(data_cleaned.columns), 'survived', columna_a_mover)

# Modificar 'embarked_id' del data_raw a int
data_cleaned['embarked_id'] = data_cleaned['embarked_id'].astype(int)

# Modificar 'title_id' del data_test a int
data_test['title_id'] = data_test['title_id'].astype(int)

data_cleaned.to_csv(r'C:\Users\Pato\Desktop\VS\DS\Titanic\data_cleaned', index=False)
data_test.to_csv(r'C:\Users\Pato\Desktop\VS\DS\Titanic\data_test', index=False)
