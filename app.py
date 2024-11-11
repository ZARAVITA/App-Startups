# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:39:47 2024

@author: ZARAVITA Haydar
"""

#REGRESSION LINEAIRE MULTIPLES

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv("50_Startups.csv")
#print(df.head())



x=df.iloc[:, :-1]
y=df.iloc[:,-1].values


from sklearn.compose import ColumnTransformer  #permet de transformer séparement les colonnes
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encorder', OneHotEncoder(),[3])], remainder="passthrough") #3 est l'index sur colonne de state, remainder est le reste, passthrough: ne change pas ces colonnes restantes

X=np.array(ct.fit_transform(x))
#print(type(X))                        #<class 'numpy.ndarray'>
#print(X)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=45)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


#APPRENTISSAGE
regressor.fit(x_train, y_train)
#Prediction
y_pred=regressor.predict(x_test)


#COmparaison
daf=pd.DataFrame({"Real values": y_test, "predicted values": y_pred})

#print(daf.head(10))


#EVALUATION DU MODELE
#CORRELATION
print("La correlation entre les deux:", daf.corr().iloc[1,0])    
#La correlation entre les deux: 0.9823056514706142

#erreur quadratique moyenne
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))   #9074.591079647453

#R&D Spend  Administration  Marketing Spend    Profit
val_predi=pd.DataFrame({"R&D Spend":[165349.20], "Administration": [136897.80], "Marketing Spend":[471784.10], "State": ["New York"]})
#pd.DataFrame({"R&D Spend":np.array([165349.20]), "Administration": np.array([136897.80]), "Marketing Spend":np.array([471784.10]), "State": ["New York"]})
v_transformed=np.array(ct.transform(val_predi))

print("La prédicion: ",regressor.predict(v_transformed))


#MISE EN PRODUCTION DU MODELE
import streamlit as st
# Application Streamlit
st.title("Prédiction de Profit pour Startups")
#DESCRIPTION
st.markdown("""
Cette application permet de prédire le bénéfice d'une startup en fonction de ses dépenses dans quatres domaines clés — **Recherche et Développement (R&D)**, **Marketing** et **Administration** (frais administratifs, salaires, etc.) — ainsi que de sa **région d'implantation**. Grâce à un modèle de régression multiple basé sur le Machine Learning (ML) en Python, l'application atteint une précision de **96 %** pour fournir des prévisions fiables.

Conçue pour répondre aux besoins des entrepreneurs et des investisseurs, cette application aide non seulement les startups à optimiser leurs budgets, mais surtout les investisseurs à identifier les startups les plus rentables. Elle peut également être utile aux analystes financiers et aux conseillers en investissement qui recherchent des indicateurs de performance basés sur des données concrètes.  

  
**Limite du Modèle**: Veuillez noter que ce modèle inclut une valeur de base (intercept) pour prédire le profit même lorsque les dépenses sont nulles. Par conséquent, les prédictions pourraient montrer un profit minimum artificiel même sans investissements en R&D, marketing, ou administration, ce qui reflète une simplification du modèle.  
> **Note** : Cette application a été développée dans le cadre d’un portfolio et repose uniquement sur les données de 50 startups. Elle ne constitue pas une solution fiable pour une application concrète, et l’auteur décline toute responsabilité quant à son usage à des fins décisionnelles.  

**Projet DATA SCIENCE développé par ZARAVITA**  
E-mail : zaravitamds18@gmail.com  
LinkedIn: http://linkedin.com/in/zara-vita-567464202
""")

# Entrées utilisateur pour les caractéristiques
R_D_Spend = st.number_input("Dépenses en R&D :", min_value=0.0, step=1000.0)
Administration = st.number_input("Administration :", min_value=0.0, step=1000.0)
Marketing_Spend = st.number_input("Dépenses Marketing :", min_value=0.0, step=1000.0)
State = st.selectbox("État :", ["Casablanca-Settat", "Tanger-Tetouan-AlHoceima", "Rabat-Salé-Kenitra"])
#CONTEXTE MAROC
if State=="Casablanca-Settat":
    State="New York"
elif State == "Tanger-Tetouan-AlHoceima" :
        State="California"
else : 
    State="Florida"    

# Préparer les données pour la prédiction
if st.button("Prédire"):
    input_data = pd.DataFrame({
        "R&D Spend": [R_D_Spend],
        "Administration": [Administration],
        "Marketing Spend": [Marketing_Spend],
        "State": [State]
    })
    input_transformed = np.array(ct.transform(input_data))

    # Faire la prédiction
    prediction = regressor.predict(input_transformed)
    st.write(f"Profit prédit : {round(prediction[0], 2)} Dhs par an")














