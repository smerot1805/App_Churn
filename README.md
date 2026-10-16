# App_Churn
Application de prédiction du churn à partir de l'âge, du nombre de sites, de la présence d'un account manager, du total des achats et de l'ancienneté.

## Lancer l'application

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Le modèle de régression logistique est entraîné au démarrage sur `data/customer_churn.csv`. Il utilise uniquement les cinq variables affichées dans le formulaire.
