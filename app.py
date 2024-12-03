# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.ML.Descriptors import MoleculeDescriptors
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu # for setting up menu bar


# +
#-----------Web page setting-------------------#
page_title = "💊Breast Cancer pIC50 Prediction Web App"
page_icon = "🎗🧬⌬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"
#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

# Title of the app
#st.title("pIC50 Prediction App")
# Logo image
image = 'logo/logo.jpg'
st.image(image, use_column_width=True)

#https://github.com/afolabiowoloye/fff/blob/main/logo.jpg

# -

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'EGFR', 'ER','Prog', 'About'],
    icons = ["house-fill", "capsule", "heart-fill", "heart", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

# +
if selected == "Home":
    st.subheader("Welcome to Breast Cancer pIC50 Prediction Web App")
    st.write("This application is designed to assist researchers and healthcare professionals in predicting the half-maximal inhibitory concentration (IC50) values for various compounds in the treatment of breast cancer. Understanding IC50 values is crucial for evaluating the effectiveness of therapeutic agents and optimizing treatment strategies")
    
    
# Display data preview
#st.write("Data Preview:")
#st.dataframe(df.head())
if selected == "EGFR":
# Link to the dataset on Google Drive
    data_link_id = "1C-cFzESEfJcEdWGDLLdVawQbgZvbU_rQ"
    data_link = f'https://drive.google.com/uc?id={data_link_id}'
    data = pd.read_csv(data_link)
    st.write("Data Preview:")
    st.dataframe(data.head())
    #col = data.columns
    #st.write(col)

        # Data preprocessing
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)

    X = data.drop('pIC50', axis=1)
    y = data['pIC50']

        # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'Mean Squared Error: {mse:.2f}')
    st.write(f'Root Mean Squared Error: {rmse:.2f}')
    st.write(f'R-squared (R2) Score: {r2:.2f}')

    # Plotting
    fig, ax = plt.subplots()
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.3, 'color': 'blue'}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel('Experimental pIC50')
    ax.set_ylabel('Predicted pIC50')
    ax.set_xlim(2, 11.5)
    ax.set_ylim(2, 11.5)
    plt.title('Actual vs Predicted pIC50')
    st.pyplot(fig)

# File uploader for SMILES data
    smiles_file = st.file_uploader("Upload your sample.csv", type="csv")
    st.sidebar.markdown("""[Example input file](https://raw.githubusercontent.com/afolabiowoloye/xyz/refs/heads/main/sample.csv)
""")
    if smiles_file is not None:
        sample = pd.read_csv(smiles_file)
        st.write("Sample Data Preview:")
        st.dataframe(sample.head())


    # Getting RDKit descriptors
        def RDKit_descriptors(SMILES):
            mols = [Chem.MolFromSmiles(i) for i in SMILES]
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            desc_names = calc.GetDescriptorNames()
            Mol_descriptors = []
            for mol in mols:
                mol = Chem.AddHs(mol)
                descriptors = calc.CalcDescriptors(mol)
                Mol_descriptors.append(descriptors)
            return Mol_descriptors, desc_names

        MoleculeDescriptors_list, desc_names = RDKit_descriptors(sample['SMILES'])
        df_ligands_descriptors = pd.DataFrame(MoleculeDescriptors_list, columns=desc_names)
        st.dataframe(df_ligands_descriptors.head())
        
        df_ligands_descriptors = df_ligands_descriptors.drop(["SPS", "AvgIpc"], axis=1)
        #col2 = df_ligands_descriptors.columns
        #st.write(col2)

    # Combine descriptors


    # Predictions
        sample['predicted_pIC50'] = model.predict(df_ligands_descriptors)
        st.write("Predicted pIC50 Values:")
        st.dataframe(sample[['SMILES', 'predicted_pIC50']])
        download_result = pd.DataFrame(sample)
        download_result = download_result.to_csv(index=False)
        st.download_button("Press to Download Result",download_result,"file.csv","text/csv",key='download-csv')

# -

