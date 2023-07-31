#!/usr/bin/env python
# coding: utf-8

# Goal: using the model to predict HI in MG
# Author: Danieli M. F.
# Date: 28/06/23

# -------------------------   
# Bibliotecas
# -------------------------   

import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt
import numpy             as np
import plotly.express as px
import plotly.io as pio
import inflection
import streamlit as st
import folium
import pickle
# import joblib
import geopandas as gpd
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from IPython.display     import display, HTML
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Water quality',page_icon='💦', layout='wide')

# -------------------------   
# Data
# -------------------------   

# with open('df_aux1.pickle', 'rb') as file:
#     df = pickle.load(file)
df = pd.read_csv('df_aux1.csv')

# criar abas
tab1,tab2 = st.tabs(['ℹ️ How to use this app','✓ Model Application'])

with tab1: 
    with st.container(): 

        st.subheader('Health risk to drink river water in Minas Gerais')
        
        st.markdown('##### How to use this app')
         
        st.write('In the sidebar, you can alter water characteristics and set a factor of change for land cover across the state. A tradeoff is set between forest and the other uses (if urban area, agriculture, or mining increase, it is assumed that they replace forest cover). It must be stated that the model accounts only for change in the land cover area. This means that the increase/decrease of diffuse or point load input from these areas is not accounted. Notheless, these affects may be assessed varying other variables, such as organic matter and nutrient content within the rivers.')
        st.write('Further information about data and methods are presented in..link artigo..')
        st.write('In the tab ***"Model Application"*** you will verify the predicted risk to human health due to long-exposure to river water ingestion, represented by the Health Index.')

        st.write('The risk is calculated as a Health Quotient:')

        st. write("""
        > HQ = CDI/RfD
        >
        > RfD: represents the maximum amount of metal that potentially causes no chronic effects (mg/kg/d) 
        >
        > CDI = (C x IR x EF x ED) / (BW x AT)
        >
        > C is the iron or manganese concentration in water (mg/L); 
        > IR is the human water ingestion rate in L/day (2.2 L/day for adults);
        > ED is the exposure duration in years (70 years for adults); 
        > EF is the exposure frequency in days/year (365 days for adults); 
        > BW is the average body weight in kg (70 kg for adults); 
        > AT is the averaging time (AT = 365 × ED).
        >
        > The Health Index (HI) is the sum of HQ for each metal. A HI > suggests a possible risk for non-carcinogenic effects.
        """)          

# ******************
# Barra lateral Streamlit
# ******************  

st.sidebar.markdown('# Risk to health of adults due to long-term river water ingestion in Minas Gerais')

st.sidebar.markdown("""---""")

st.sidebar.markdown('### Change the variables')

# loading model
model = pickle.load( open( 'BayesSearchCV_theilsen.pkl', 'rb') )

# ******************
# Helper functions
# ******************  
# Função para encoding

def tratamento_encoding(X):

    df_tratado = X.copy() 
    
    ### 7.2 Encoding
    # - condição_de_tempo: ordinal encoding (numera em ordem hierárquica)
    # - season: one hot encoding -> 0 (dry) e 1 (rainy)
    # - bacia hidrográfica, estação, curso_dágua e upgrh_sigla: label encoding -> dá valores inteiros aleatórios para cada uma 

    dicionario = {'Bom':1, 'Nublado':2, 'Chuvoso':3}
    df_tratado['weather_condition'] = df_tratado['weather_condition'].map( dicionario )

    df_tratado.loc[df_tratado['period'] == 'Dry', 'period'] = 0
    df_tratado.loc[df_tratado['period'] == 'Rainy', 'period'] = 1
    df_tratado['period'] = df_tratado['period'].astype(int)

    le = LabelEncoder() 
    df_tratado['watershed'] = le.fit_transform(df_tratado['watershed']) 

    le1 = LabelEncoder()
    df_tratado['station'] = le1.fit_transform(df_tratado['station'])

    le2 = LabelEncoder()
    df_tratado['acronym'] = le2.fit_transform(df_tratado['acronym'])

    le3 = LabelEncoder()
    df_tratado['watercourse'] = le3.fit_transform(df_tratado['watercourse'])
    
    return df_tratado

# função para receber dados do usuário e juntar com outras features
def input_data():
    # USER DATA
    total_phosphorus = st.sidebar.slider('Total phosphorus (mg/L)', 0.1, 10.0, 0.1)
    BOD = st.sidebar.slider('BOD (mg/L)', 0.1, 10.0, 1.0)
    ph = st.sidebar.slider('pH', 5.15, 12.0, 7.0)
    temperature = st.sidebar.slider('Temperature (˚C)', 7.5, 31.0, 22.0)
    do = st.sidebar.slider('Dissolved oxygen (mg/L)', 2.0, 9.5, 8.0)
    turbidity = st.sidebar.slider('Turbidity (NTU)', 10.0, 1000.0, 100.0)
    conductivity = st.sidebar.slider('Conductivity ', 5.0, 400.0, 100.0)
    suspended_solids = st.sidebar.slider('Suspended solids', 1.0, 1000.0, 100.0)
    agriculture_fact = st.sidebar.slider('Agriculture', 0.1, 1.5, 1.0) 
    mining_fact = st.sidebar.slider('Mining', 0.1, 1.5, 1.0) 
    urban_fact = st.sidebar.slider('Urban', 0.1, 1.5, 1.0)  

    # um dicionário recebe as informações acima
    user_data = {'agriculture': agriculture_fact, 
                 'mining': mining_fact, 
                 'urban_infrastructure': urban_fact,
                 'pH': ph,
                 'total_phosphorus': total_phosphorus,
                 'BOD': BOD,
                 'temperature': temperature,
                 'dissolved_oxygen': do,
                 'turbidity': turbidity,
                 'in-situ_electrical_conductivity': conductivity,
                 'total_suspended_solids': suspended_solids,
                 }
   
    num_rows = len(df)  
    user_dataframe = pd.DataFrame([user_data] * num_rows)
    user_dataframe = user_dataframe.loc[:, 'pH':]
    
    # outras features pre-definidas, mas alteradas pelo usuário
    dados_pre = df.drop(['turbidity', 'pH', 'temperature', 'dissolved_oxygen', 
                         'total_phosphorus','BOD',
                         'in-situ_electrical_conductivity', 'total_suspended_solids', 
                         'forest', 'mining', 'urban_infrastructure',
                         'agriculture',
                         'health_index'], axis=1) 

    # Multiplicando as áreas de uso do solo pelo fator selecionado pelo user
    dados_pre['agriculture'] = df['agriculture']*agriculture_fact
    dados_pre['mining'] = df['mining']*mining_fact 
    dados_pre['urban_infrastructure'] = df['urban_infrastructure']*urban_fact 
    
    # trade-off entre usos alterados e floresta
    
    if agriculture_fact >= 1:
        direction1 = 1  # Increase the values
    else:
        direction1 = -1  # Decrease the values    
    if mining_fact >= 1:
        direction2 = 1 
    else:
        direction2 = -1         
    if urban_fact >= 1:
        direction3 = 1 
    else:
        direction3 = -1 
        
    dados_pre['forest'] = df['forest'] - df['agriculture'] * (1 - agriculture_fact) * direction1 - dados_pre['mining'] * (1 - mining_fact) * direction2 - df['urban_infrastructure'] * (1 - urban_fact) * direction3
     
    # Concatenate the modified features and predefined features
    dados = pd.concat([dados_pre], axis=1).reset_index(drop=True) 
    features = pd.concat([dados, user_dataframe.reset_index(drop=True)], axis=1)

    # Corringindo a ordem das colunas 
    desired_order = ['altitude', 'watershed', 'watercourse', 'station',
                 'decimal_latitude', 'decimal_longitude', 'total_alkalinity',
                 'weather_condition', 'in-situ_electrical_conductivity',
                 'true_color', 'nitrate', 'nitrite', 'total_dissolved_solids',
                 'total_suspended_solids', 'sulfide', 'air_temperature', 'year',
                 'month', 'day', 'day_of_week', 'week_of_year', 'acronym',
                 'agriculture', 'forest', 'mining', 'non_observed',
                 'other_non_forest_natural_formation', 'other_non_vegetated_area',
                 'urban_infrastructure', 'water', 'period', 'BOD',
                 'total_phosphorus', 'turbidity', 'pH', 'temperature',
                 'dissolved_oxygen', 'river_flow']

    features = features.reindex(columns=desired_order)

    return features
    
# form the new input dataset (X_test)
user_input_variables = input_data()                                              
X_new = tratamento_encoding(user_input_variables)

# New prediction
prediction = model.predict(X_new)

# Replace negative predictions with zero
prediction = np.maximum(prediction, 0.0001)  # TAVA GERANDO ALGUNS HI NEGATIVOS

with tab2: 
    # média HI simulados e boxplot em todo o estado
    with st.container(): 
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('##### 📊 Mean predicted Health Index')
            
        with col2:
            col2.metric(' ', np.round(np.mean(prediction),2))
    
    
        st.markdown('##### 🔎 Box Plot with Health Index across the state')
        fig = px.box(prediction, width = 500, height = 200, orientation='h')
        fig.update_layout(xaxis_title=' ', yaxis_title=' ')  

        st.plotly_chart(fig)
            
        # mapa de HI simulado por estação
        st.markdown('##### 📌 Map of predicted Health Index')
        
        # Create a DataFrame with latitude, longitude, and predictions
        prediction_df = pd.DataFrame({'latitude': df['decimal_latitude'],
                                      'longitude': df['decimal_longitude'],
                                      'prediction': prediction, 
                                      'station': df['station']})

        # Generate the map centered on a specific location
        m = folium.Map(location=[-18.3, -44], zoom_start=6)

        # Define a colormap for the predictions
        colormap = folium.LinearColormap(
            colors=['green','navy','tomato'],#'lavenderblush', 'darkmagenta', 'indigo'],  
            vmin=prediction_df['prediction'].min(),
            vmax=prediction_df['prediction'].max()
        )

        # Add circles to the map for each prediction
        for index, row in prediction_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            value = row['prediction']
            estacao = row['station']

            # Create a circle marker with size and color based on the prediction value
            circle = folium.CircleMarker(
                location=[lat, lon],
                radius=5 * value,  
                color=colormap(value),  
                fill=True,
                fill_color=colormap(value),
                fill_opacity=0.3,
                weight=0.2
            )

            # Create a popup with the "estação" name
            popup = folium.Popup(estacao, max_width=250)

            # Add the popup to the circle marker
            circle.add_child(popup)

            # Add the circle marker to the map
            circle.add_to(m)

        # Add the colormap to the map
        colormap.add_to(m)

        # Display the map in Streamlit
        folium_static(m)


####################################################################################### ISSO ABAIXO SERVE PARA COLOCAR UMA BARRA DE ACOMPANHAMENTO - FUNCIONOU, MAS ESCOLHI NAO USAR AQUI

#     with st.container():

#         import streamlit as st

#         # Function to customize the CSS of the slider
#         def set_slider_style():
#             st.markdown(
#                 """
#                 <style>
#                 .slider-content {
#                     margin-top: 20px;
#                 }

#                 .slider-value {
#                     float: right;
#                 }

#                 .risk-bar {
#                     background-color: #ddd;
#                     height: 20px;
#                     border-radius: 10px;
#                     margin-top: 10px;
#                 }

#                 .risk-value {
#                     background-color: #ff4d4d;
#                     height: 100%;
#                     border-radius: inherit;
#                     width: 0;
#                     transition: width 0.3s;
#                 }
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )

#         # Function to render the risk bar
#         def render_risk_bar(risk_level, compare_value):
#             risk_percentage = risk_level / compare_value * 10
#             st.markdown('<div class="risk-bar"><div class="risk-value" style="width:{}%;"></div></div>'.format(risk_percentage), unsafe_allow_html=True)

#         # Set the style for the slider
#         set_slider_style()

#         # Calculate the risk level
#         calculated_risk_level = np.round(np.mean(prediction),2)
#         compare_value = 1

#         # Render the risk bar
#         render_risk_bar(calculated_risk_level, compare_value)

#         # Display the risk level and comparison value
#         st.markdown(f"Mean Risk Level: {calculated_risk_level}")
#         # st.markdown(f"Comparison Value: {compare_value}")


       
st.sidebar.markdown("""---""")

st.sidebar.markdown('#### This app is under development. Contact danimf15@hotmail.com for more details.')
