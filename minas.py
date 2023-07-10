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
import geopandas as gpd
import pickle

from streamlit_folium import folium_static        
from IPython.display     import display, HTML
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import pearsonr

from scipy.optimize import fsolve
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
# from pxmap import px_static
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder

st.set_page_config(page_title='Water quality',page_icon='💦', layout='wide')

# -------------------------   
# Data
# -------------------------   
# import joblib
# df = joblib.load('df_aux1.pkl')

# with open('df_aux1.pickle', 'rb') as file:
#    df = pickle.load(file)

df = pd.read_csv('df_aux1.csv')  

# df = pd.read_excel('df_aux1.xlsx').drop('Unnamed: 0',axis=1)
# numeric_features = df.select_dtypes(include=['int', 'float']).columns
# df[numeric_features] = df[numeric_features].astype(float)

# st.header('Modeling Fe and Mn during the dry and rainy')

# criar abas
tab1,tab2 = st.tabs(['How to use this app','Model Application'])

with tab1: 
    with st.container(): 
        # st.subheader('About')

        st.subheader('Health risk to drink river water in Minas Gerais')
        
        st.markdown('##### How to use this app')
         
        st.write('In the sidebar, you can alter water characteristics and set a factor of change for land cover across the state. A tradeoff is set between forest and the other uses (if urban area, agriculture, and mining increase, it is assumed that they replace forest cover.')
        st.write('In the tab ***"Model Application"*** you will verify the predicted risk to human health due to river water ingestion.')

        st.write('The risk is calculated as a Health quotient:')
    

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
        
        #st.write('The tab ***"Data"*** summarizes the observed data used for model development, while the tab ***"Model information"*** presents the model performance (under developmnent).')
        
            

# ******************
# Barra lateral Streamlit
# ******************  

st.sidebar.markdown('# Health index due to river water ingestion in Minas Gerais')
# st.sidebar.markdown('## Water quality in rivers')
# st.sidebar.image("hidro.gif")

st.sidebar.markdown("""---""")


st.sidebar.markdown('### Change the variables')


# loading model
# model = pickle.load( open( '../forest_GridSearchCV.pkl', 'rb') )
# model = pickle.load( open( '../pkl/linear_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/knn_model.pkl', 'rb') )

model = pickle.load( open( 'knn_model.pkl', 'rb') )

# model = pickle.load( open( '../pkl/lasso_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/ann_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/ridge_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/svr_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/huber_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/rforest_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/xgb_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/adaboost_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/catboost_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/lgbm_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/theil.pkl', 'rb') )
# model = pickle.load( open( '../pkl/ransac_model.pkl', 'rb') )
# model = pickle.load( open( '../pkl/extra_model.pkl', 'rb') )

# ******************
# Helper functions
# ******************  
# SOMENTE ENCODING

def tratamento_encoding(X):

    df_tratado = X # pd.concat([X,Y], axis=1)
    
    ### 7.2 Encoding
    # - condição_de_tempo: ordinal encoding (numera em ordem hierárquica) >> TESTAR OUTROS DEPOIS
    # - season: one hot encoding -> 0 (dry) e 1 (rainy)
    # - bacia hidrográfica, estação, curso_dágua e upgrh_sigla: label encoding -> dá valores inteiros aleatórios para cada uma >> DEPOIS PROCURAR ALGUM MÉTODO COM BASE EM PESOS SEGUNDO A ÁREA DE DRENAGEM, P.EX.

    dicionario = {'Bom':1, 'Nublado':2, 'Chuvoso':3}
    df_tratado['condição_de_tempo'] = df_tratado['condição_de_tempo'].map( dicionario )

    df_tratado.loc[df_tratado['period'] == 'Dry', 'period'] = 0
    df_tratado.loc[df_tratado['period'] == 'Rainy', 'period'] = 1
    df_tratado['period'] = df_tratado['period'].astype(int)

    le = LabelEncoder() # Cria um objeto LabelEncoder
    df_tratado['bacia_hidrográfica'] = le.fit_transform(df_tratado['bacia_hidrográfica']) # Aplica o Label Encoding aos dados

    le1 = LabelEncoder()
    df_tratado['estação'] = le1.fit_transform(df_tratado['estação'])

    le2 = LabelEncoder()
    df_tratado['sigla'] = le2.fit_transform(df_tratado['sigla'])

    le3 = LabelEncoder()
    df_tratado['curso_dágua'] = le3.fit_transform(df_tratado['curso_dágua'])
    
     # day of week
    df_tratado['year_sin'] = df_tratado['year'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
    df_tratado['year_cos'] = df_tratado['year'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

    # month
    df_tratado['month_sin'] = df_tratado['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
    df_tratado['month_cos'] = df_tratado['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

    # day 
    df_tratado['day_sin'] = df_tratado['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    df_tratado['day_cos'] = df_tratado['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )
    
    # others
    df_tratado['day_of_week_sin'] = df_tratado['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    df_tratado['day_of_week_cos'] = df_tratado['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )
    df_tratado['week_of_year_sin'] = df_tratado['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
    df_tratado['week_of_year_cos'] = df_tratado['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )


    return df_tratado


def input_data():
    # USER DATA
    # os valores sao min, max e med do dataset observado
    phosphorus_total = st.sidebar.slider('Total phosphorus', 0.1, 10.0, 0.1)
    BOD = st.sidebar.slider('BOD', 0.1, 10.0, 1.0)
    ph = st.sidebar.slider('pH', 5.15, 12.0, 7.0)
    temperature = st.sidebar.slider('Temperature', 7.5, 31.0, 25.0)
    do = st.sidebar.slider('Dissolved oxygen (mg/L)', 2.0, 9.5, 7.0)
    turbidity = st.sidebar.slider('Turbidity', 10.0, 1000.0, 100.0)
    conductivity = st.sidebar.slider('Conductivity', 5.0, 400.0, 100.0)
    suspended_solids = st.sidebar.slider('Suspended solids', 1.0, 1000.0, 100.0)
    dissolved_solids = st.sidebar.slider('Dissolved solids', 1.0, 1000.0, 100.0)
    agriculture_fact = st.sidebar.slider('Agriculture', 0.1, 1.5, 1.0) 
    mining_fact = st.sidebar.slider('Mining', 0.1, 1.5, 1.0) 
    urban_fact = st.sidebar.slider('Urban', 0.1, 1.5, 1.0) 

    # um dicionário recebe as informações acima
    user_data = {'agriculture': agriculture_fact, 
                 'mining': mining_fact, 
                 'urban infrastructure': urban_fact,
                 'pH': ph,
                 'phosphorus_total': phosphorus_total,
                 'BOD': BOD,
                 'temperature': temperature,
                 'dissolved_oxygen': do,
                 'turbidity': turbidity,
                 'condutividade_elétrica_in_loco': conductivity,
                 'sólidos_em_suspensão_totais': suspended_solids,
                 'sólidos_dissolvidos_totais': dissolved_solids,
                 }
   
    num_rows = len(df)  
    user_dataframe = pd.DataFrame([user_data] * num_rows)
    user_dataframe = user_dataframe.loc[:, 'pH':]
    
    # outras features pre-definidas
    dados_pre = df.drop(['turbidity', 'pH', 'temperature', 'dissolved_oxygen', 
                         'phosphorus_total','BOD',
                         'condutividade_elétrica_in_loco', 'sólidos_em_suspensão_totais', 
                         'sólidos_dissolvidos_totais', 'forest', 'mining', 'urban infrastructure',
                         'agriculture',
                         'health_index'], axis=1) # dropping as features que vão se alterar 

    # Multiply the columns in 'df' with the corresponding percentages
    dados_pre['agriculture'] = df['agriculture']*agriculture_fact
    dados_pre['mining'] = df['mining']*mining_fact 
    dados_pre['urban infrastructure'] = df['urban infrastructure']*urban_fact 
    
    # trade-off: 
    # determine the direction of alteration based on the factor
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
        
    dados_pre['forest'] = df['forest'] - df['agriculture'] * (1 - agriculture_fact) * direction1 - dados_pre['mining'] * (1 - mining_fact) * direction2 - df['urban infrastructure'] * (1 - urban_fact) * direction3
                  
    
#     # trade-off of land areas to keep the total area of the state
#     #- urban and mining areas with 'other non vegetated area' 
#     old_areas = df[['mining', 'urban infrastructure']]
#     new_areas = old_areas.apply(lambda x: x * user_data[x.name] if isinstance(user_data[x.name], (int, float)) else x)
#     total_difference = (old_areas - new_areas).sum().sum()
#     df['other non vegetated area'] = df['other non vegetated area'] - total_difference
    
    
    # Concatenate the modified features and predefined features
    dados = pd.concat([dados_pre], axis=1).reset_index(drop=True) 
    features = pd.concat([dados, user_dataframe.reset_index(drop=True)], axis=1)

   #  st.write(features[['agriculture', 'forest', 'mining',
   # 'non observed', 'other non forest natural formation',
   # 'other non vegetated area', 'urban infrastructure', 'water','year']])
    
    return features
    
# form the new input dataset (X_test)
user_input_variables = input_data()                                              
X_new = tratamento_encoding(user_input_variables)

st.write(X_new) 

# New prediction
prediction = model.predict(X_new)

# st.write(user_input_variables.mean())


   
with tab2: 
    # with st.container(): 
    #     st.subheader('Predicted HI:')


#         st.subheader('Model application')

#         st.write('Input water characteristics:')

#         # st.dataframe(user_input_variables)
#         fig = px.bar(user_input_variables,barmode='group', color_discrete_sequence=[
#                      "forestgreen","coral", "darkcyan", "firebrick", "dimgray", "blue",
#                      "violet","hotpink","blueviolet","crimson","slateblue","indigo", "midnightblue"],
#                      labels={"value": "Value",
#                      "index": " ",
#                  })
        # st.plotly_chart(fig,use_container_width=True)

    with st.container(): 
        
        col1, col2 = st.columns([1,2])
        
        with col1:
            st.markdown('###### Mean predicted Health Index')
            col1.metric(' ', np.round(np.mean(prediction),2))#, delta=str(1),delta_color="inverse")
            
        with col2:
            st.markdown('###### Box Plot with Health Index across the state')
            fig = px.box(prediction, width = 500, height = 200, orientation='h')
            fig.update_layout(xaxis_title=' ', yaxis_title=' ')  # Set the desired labels

            st.plotly_chart(fig)
#         with col2:
#             col1.metric('Fe: rainy', np.round(prediction,2),delta=str(np.round(prediction-0.3,2)).replace('[','').replace(']',''),delta_color="inverse")
        
            
    # with st.container():
#         st.markdown("""
# > *The number accompanied by the arrow indicates the difference between the predicted and the limit concentration. 
# > The red color indicates the mg/L by which the predicted concentration is larger than the limit; the green is the contrary.*
#         """)
#         # resultados_vs_limite = [[float(prediction_chuvoso_Fe),0.3],[float(prediction_seca_Fe),0.3],
#         #                         [float(prediction_chuvoso_Mn),0.1],[float(prediction_seca_Mn),0.1]]
#         # dados = pd.DataFrame([[float(prediction_seca_Mn),0.1]], columns=['Predicted','Limit'])
#         # fig = px.bar(resultados_vs_limite,barmode='overlay',opacity=0.9, text_auto=True,
#         #  labels={"value": " ",
#         #  "index": " ",
#         # })
#         # # fig.update_layout(showlegend=False)
#         # st.plotly_chart(fig,use_container_width=True)
            
#         # health = pd.DataFrame([health_index_seca,health_index_chuvoso])
#         # health.index =['Dry','Rainy']
#         # fig = px.bar(health,labels={
#                  #     "value": "Health Index (HI)",
#                  #     "index": " ",            
#                  # })
#         # fig.update_layout(showlegend=False)
#         # st.plotly_chart(fig)
#         st.markdown("""
# > A HI > 1 suggests a possible risk""")
        
#         # fig = px.choropleth(locationmode="USA-states", color=[1], scope="usa")
#         # st.plotly_chart(fig,use_container_width=True)

       #######################################################################################    

        import folium
        from folium.plugins import MarkerCluster
        from streamlit_folium import folium_static
        import pandas as pd

        # Create a DataFrame with latitude, longitude, and predictions
        prediction_df = pd.DataFrame({'latitude': df['latitude_graus_decimais'],
                                      'longitude': df['longitude_graus_decimais'],
                                      'prediction': prediction, #df['health_index'],
                                      'estação': df['estação']})

        # Generate the map centered on a specific location
        m = folium.Map(location=[-18.3, -44], zoom_start=6)

        # Define a colormap for the predictions
        colormap = folium.LinearColormap(
            colors=['lavenderblush', 'darkmagenta', 'indigo'],  
            vmin=prediction_df['prediction'].min(),
            vmax=prediction_df['prediction'].max()
        )

        # Add circles to the map for each prediction
        for index, row in prediction_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            value = row['prediction']
            estacao = row['estação']

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
            popup = folium.Popup(prediction, max_width=250)

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
