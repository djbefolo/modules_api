import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

###---------- load data prob -------- 
def load_prob_data(sample_size):
    dataset_exported = pd.read_csv("data/dataset_exported.csv", nrows=sample_size)
    return dataset_exported

###---------- load data -------- 
def load_all_data(sample_size):
    
    data = pd.read_csv("data/df_final.csv", nrows=sample_size)

    #data = df.drop("Unnamed: 0", axis=1)

    y_pred_test_export = pd.read_csv("data/y_pred_test_export.csv")

    # Preparation des données age
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'] / 365
    data['age_bins'] = pd.cut(data['DAYS_BIRTH'], bins=np.linspace(20, 70, num=11))
    #data['age_bins'] = (pd.cut(data['DAYS_BIRTH'], bins=bins)).astype(str)

    # Charger le dataframe 'application_train.csv'
    file_path = os.path.abspath('data/application_train.csv')
    train_set = pd.read_csv(file_path, nrows=sample_size)
    #train_set = pd.read_csv('data\application_train.csv', nrows=sample_size)


    return data, y_pred_test_export, train_set

# ------------- Affichage des infos client en HTML ------------------------------------------
def display_client_info(id, revenu, age, nb_ann_travail, jours_employe):
    components.html(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous">
        </script>
        <div class="card" style="width: 500px; margin:10px;padding:0">
            <div class="card-body">
                <h5 class="card-title">Info Client</h5>

                <ul class="list-group list-group-flush">
                    <li class="list-group-item"> <b>ID                           : </b>""" + id + """</li>
                    <li class="list-group-item"> <b>Revenu                       : </b>""" + revenu + """</li>
                    <li class="list-group-item"> <b>Age                          : </b>""" + age + """</li>
                    <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>""" + nb_ann_travail + """</li>
                    <li class="list-group-item"> <b>Jours employé                : </b>""" + jours_employe + """</li>
                </ul>
            </div>
        </div>
        """,
        height=300
    )


def main():
    st.title("Mon Application Streamlit")

    # Charger les données
    data, y_pred_test_export, train_set = load_all_data(sample_size=10)  # Utilisez la taille d'échantillon souhaitée

    # Sélectionner un client
    selected_client_id = st.selectbox("Sélectionnez un client:", data["SK_ID_CURR"])

    # Afficher les informations du client sélectionné
    if selected_client_id:
        client_info = data[data["SK_ID_CURR"] == selected_client_id]
        display_client_info(
            str(client_info["SK_ID_CURR"].values[0]),
            str(client_info["AMT_INCOME_TOTAL"].values[0]),
            str(client_info["DAYS_BIRTH"].values[0]),
            str(client_info["age_bins"].values[0]),
            str(client_info["DAYS_EMPLOYED"].values[0])
        )

if __name__ == "__main__":
    main()
