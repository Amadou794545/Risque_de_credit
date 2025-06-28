import streamlit as st
import pandas as pd
import numpy as np
import joblib as jl
import plotly.express as px

#titre
st.title("Risque de Crédit")
st.write("Ce projet utilise un modèle de machine learning pour prédire le risque de crédit d'un client.")
# Explication
st.write("Le modèle prédit si un client est susceptible de faire défaut sur un prêt en fonction de ses caractéristiques personnelles et financières.")
st.write("Il est conçu pour évaluer le risque de crédit en se basant sur des données anonymisées et agrégées, garantissant ainsi la confidentialité des informations personnelles.")
st.write("L'objectif est de fournir une évaluation objective du risque de crédit, permettant aux institutions financières de prendre des décisions éclairées sur l'octroi de prêts.")


#st.sidebar
st.sidebar.title("Paramètres du Client")
# Paramètres du client
person_age = st.sidebar.number_input("Âge du Client", 18, 60,30)
person_income = st.sidebar.number_input("Revenu du Client (en année)", 1000, 10000000, 50000)
person_home_ownership = st.sidebar.selectbox("Propriété du Client", ["RENT(Locataire)", "OWN(Propriétaire)", "MORTGAGE(Hypothèque)", "OTHER(Autre)"])
person_emp_length = st.sidebar.number_input("Ancienneté de l'Emploi (en année)", 0, 40, 5)
loan_intent = st.sidebar.selectbox("Intention du Prêt", ["PERSONAL(Personnel)", "EDUCATION(Éducation)", "MEDICAL(Médical)", "DEBT_CONSOLIDATION(Consolidation de Dettes)"])
loan_grade = st.sidebar.selectbox("Note de Crédit ( si vous savez pas quoi mettre vous pouvez mettre D)", ["A", "B", "C", "D", "E", "F", "G"] )
loan_amnt = st.sidebar.number_input("Montant du Prêt", 1000, 5000000, 15000)
loan_int_rate = st.sidebar.number_input("Taux d'Intérêt du Prêt (%)", 0.0, 30.0, 5.0)
loan_percent_income = st.sidebar.number_input("Le remboursement du prêt represente combien de pourcentage sur votre salaire? (%)", 1, 100, 15)
cb_person_default_on_file = st.sidebar.selectbox("Avez-vous deja été à defaut de paiement d'un crédit bancaire dans le passé?", ["YES(OUI)", "NO(NON)"])
cb_person_cred_hist_length = st.sidebar.number_input("Si oui, combien de temps ? (en mois)", 0, 120, 0)

# Préparation des données
data = {
    "person_age": [person_age],
    "person_income": [person_income],
    "person_home_ownership": [person_home_ownership],
    "person_emp_length": [person_emp_length],
    "loan_intent": [loan_intent],
    "loan_grade": [loan_grade],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_default_on_file": [cb_person_default_on_file],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length]
}

input_df = pd.DataFrame(data)

st.dataframe(input_df, hide_index=True)

# Chargement du modèle
pipeline = jl.load("pipeline_credit.pkl")
# Prédiction

with open("seuil.txt", "r") as f:
   seuil_optimal = float(f.read())

def make_prediction( features):
    y_probs = pipeline.predict_proba(features)
    y_pred = (y_probs >= seuil_optimal).astype(int)
    y_probs = np.round(y_probs * 100, 2)
    return y_pred, y_probs


if st.sidebar.button("Prédire"):
    y_pred, y_probs = make_prediction(input_df)

    # Affichage des résultats
    st.subheader("Résultats de la Prédiction")
    st.write(f"Le risque de faire un crédit à ce client est : {'Accepté' if y_pred[0][0] == 0 else 'Refusé car notre structure n accepte que de prendre des risque inférieur a 20%'}")
    st.write(f"Probabilité de défaut de paiement : {y_probs[0][0]}%")

    # Graphique
    fig = px.bar(x=["le client fera defaut de paiement", "le client ne fera pas defaut paiement"], y=[y_probs[0], 100 - y_probs[0]],
                 labels={'x': 'Statut', 'y': 'Probabilité (%)'}, title='Probabilité de Risque de Crédit')
    st.plotly_chart(fig)