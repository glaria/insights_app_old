import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app_functions import *
import base64
import io
import itertools
import plotly.graph_objects as go


significance_treshold = 0.1

def highlight_pvalue(row):
    """Highlights rows with P-value <= significance_treshold."""
    if float(row["P-value"]) <= significance_treshold:
        return ["background-color: lightgreen"] * len(row)
    return [""] * len(row)

def calculate_metrics(subset, kpi):
    """Calculates metrics for a specific KPI."""
    tg_acceptors = subset.loc[subset[tgcg_column] == 'target', kpi].sum()
    tg_total = len(subset.loc[subset[tgcg_column] == 'target'])
    tg_acceptance = round((tg_acceptors / tg_total)*100,2) if tg_total != 0 else 0

    cg_acceptors = subset.loc[subset[tgcg_column] == 'control', kpi].sum()
    cg_total = len(subset.loc[subset[tgcg_column] == 'control'])
    cg_acceptance = round((cg_acceptors / cg_total) * 100, 2) if cg_total != 0 else 0

    uplift = tg_acceptance - cg_acceptance
    p_value = z2p(zscore(float(tg_acceptors)/float(tg_total), float(cg_acceptors)/float(cg_total),float(tg_total), float(cg_total))) if tg_total != 0 and cg_total != 0 else None

    return [kpi, tg_acceptors, tg_acceptance, cg_acceptors, cg_acceptance, uplift, p_value]

def download_csv_link(df, filename, message="Click here to download this table"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{message}</a>'
    return href

st.set_page_config(page_title="Analysis", page_icon="📈", layout="wide",)

st.markdown("# Campaign results")

# Replace with your actual dataset and information_dataset
dataset = pd.read_csv("pages/temp/uploaded_data.csv", sep = ',')  # Replace with your dataset
information_dataset = pd.read_csv("pages/temp/user_defined_info_dataset.csv", sep = ',')  # Replace with your information_dataset

# Extract the TGCG column from the dataset
tgcg_column = information_dataset.loc[information_dataset['METATYPE'] == 'TGCG', 'COLUMN'].values[0]

# Set to lower TGCG column: TARGET-> target, Control -> control
dataset[tgcg_column] = dataset[tgcg_column].str.lower()

tgcg_counts = dataset[tgcg_column].value_counts()

# Create a pie chart with counts and percentages
fig = px.pie(tgcg_counts.reset_index(), values=tgcg_counts, names='index', title="Target vs Control Groups")

# Display the total counts and percentages in the labels
fig.update_traces(textinfo='label+percent', textfont_size=12, insidetextorientation='radial')

# Show the pie chart in the Streamlit app
st.plotly_chart(fig)

# Get all KPI columns
kpi_columns = information_dataset.loc[information_dataset['METATYPE'] == 'KPI', 'COLUMN'].values

# Calculate metrics for each KPI and store the results in a list
results = [calculate_metrics(dataset, kpi) for kpi in kpi_columns]

# Convert the list of results to a pandas DataFrame
result_df = pd.DataFrame(results, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
result_df = result_df.applymap(format_float)

# Apply highlight function
highlighted_df = result_df.style.apply(highlight_pvalue, axis=1)

st.write(highlighted_df)

# Add a download link for CSV
st.markdown(download_csv_link(result_df, "results.csv"), unsafe_allow_html=True)

# Segment fields
st.markdown(f"# Segments with significant results")
# 1. Identify the segmentation columns
segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUM_ST', 'BOOL', 'STRING'])), 
                                              'COLUMN'].values
continue_segmentation_columns = information_dataset.loc[(information_dataset['METATYPE'] == 'SF') & 
                                              (information_dataset['DATATYPE'].isin(['NUMERIC'])), 
                                              'COLUMN'].values
#st.write(continue_segmentation_columns)
# 2. For each segmentation column, get all unique values.
for seg_column in segmentation_columns:
    unique_values = dataset[seg_column].unique()

    for unique_value in unique_values:
        subset = dataset[dataset[seg_column] == unique_value]
        results = [calculate_metrics(subset, kpi) for kpi in kpi_columns]

        result_df = pd.DataFrame(results, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
        result_df = result_df.applymap(format_float)

        # Convert the 'P-value' column to numeric
        result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')

        # Filter the DataFrame to only include rows with P-value <= significance_treshold
        result_df = result_df[result_df['P-value'] <= significance_treshold]

        if not result_df.empty:
            highlighted_df = result_df.style.apply(highlight_pvalue, axis=1)
            st.markdown(f"Segment: {seg_column} = {unique_value}")
            st.write(highlighted_df)
            st.markdown(download_csv_link(result_df, f"results_{seg_column}_{unique_value}.csv"), unsafe_allow_html=True)

##seccion variables segmentacion continuas
#Para calcular los mejores segmentos (intervalos) de variables continuas (sin definir cuantiles ) primero debemos reescalar el control group para tener el mismo número de elementos que target
# Identificar las clases minoritaria y mayoritaria
minority_class = dataset[tgcg_column].value_counts().idxmin()
majority_class = dataset[tgcg_column].value_counts().idxmax()

# Separar el dataset en dos según la clase minoritaria y mayoritaria
minority_df = dataset[dataset[tgcg_column] == minority_class]
majority_df = dataset[dataset[tgcg_column] == majority_class]

# Sobremuestrear la clase minoritaria
minority_oversampled = minority_df.sample(len(majority_df), replace=True, random_state=42)

# Combina el dataframe sobremuestreado con el dataframe de la clase mayoritaria
oversampled_df = pd.concat([majority_df, minority_oversampled], axis=0)


#ahora iteramos sobre los diferentes campos numericos para los que queremos calcular los mejores segmentos
for seg_column in continue_segmentation_columns:
    for kpi in kpi_columns:
        # Crear un dataframe solo con los registros de 'target'
        target_df = oversampled_df[oversampled_df[tgcg_column] == 'target'][[tgcg_column, seg_column,kpi]]

        # Crear un dataframe solo con los registros de 'control'
        control_df = oversampled_df[oversampled_df[tgcg_column] == 'control'][[tgcg_column, seg_column,kpi]]

        # Ordenar los dataframes
        target_df = target_df.sort_values(by=seg_column)
        control_df = control_df.sort_values(by=seg_column)

        # Resetear los índices de los dataframes
        target_df.reset_index(drop=True, inplace=True)
        control_df.reset_index(drop=True, inplace=True)

        # Renombrar las columnas en control_df antes de concatenar
        control_df.columns = [col + "_control" for col in control_df.columns]

        # Ahora concatenamos los dos df
        concatenated_df = pd.concat([target_df, control_df], axis=1)

        # Crear un nuevo dataframe con las columnas especificadas
        new_df = pd.DataFrame()

        # Calcular la media de seg_column y seg_column_control, en caso que algunos valores no coincidan exactamente tomamos la media entre ambos
        #esto normalmente no tendrá efecto, siempre que tg y cg sean homogeneos
        new_df[seg_column] = (concatenated_df[seg_column] + concatenated_df[seg_column + "_control"]) / 2

        # Calcular la diferencia entre kpi y kpi_control
        new_df['granular_uplift'] = concatenated_df[kpi] - concatenated_df[kpi + "_control"]

        #ahora hay que encontrar los subintervalos que maximicen (o minimicen) la suma de granular_uplift

        granular_uplift_array = new_df['granular_uplift'].values
        max_granular_uplift, start_index, end_index = kadane_algorithm(granular_uplift_array)

    
        # Obtén los valores de inicio y fin desde 'new_df' usando los índices obtenidos
        start_value = new_df.iloc[start_index][seg_column]
        end_value = new_df.iloc[end_index][seg_column]

        # Filtra el dataframe 'dataset' para quedarte solo con las filas donde el valor en 'seg_column' está entre 'start_value' y 'end_value'
        filtered_dataset = dataset[(dataset[seg_column] >= start_value) & (dataset[seg_column] <= end_value)]

        results = [calculate_metrics(filtered_dataset, kpi)]

        result_df = pd.DataFrame(results, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
        result_df = result_df.applymap(format_float)

        # Convert the 'P-value' column to numeric
        result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')

        # Filter the DataFrame to only include rows with P-value <= significance_treshold
        result_df = result_df[result_df['P-value'] <= significance_treshold]

        if not result_df.empty:
            highlighted_df = result_df.style.apply(highlight_pvalue, axis=1)
            st.markdown(f"Segment: {seg_column} between {start_value} and {end_value}")
            st.write(highlighted_df)
            #st.markdown(download_csv_link(result_df, f"results_{seg_column}_{unique_value}.csv"), unsafe_allow_html=True)

##fin seccion variables segmentacion continuas

def calculate_relative_uplift(subset, kpi1, kpi2, value1, value2):
    tg_count = len(subset[(subset[tgcg_column] == 'target') & (subset[kpi1] == value1) & (subset[kpi2] == value2)])
    tg_total = len(subset[subset[tgcg_column] == 'target'])

    cg_count = len(subset[(subset[tgcg_column] == 'control') & (subset[kpi1] == value1) & (subset[kpi2] == value2)])
    cg_total = len(subset[subset[tgcg_column] == 'control'])

    tg_acceptance = round((tg_count / tg_total) * 100, 2) if tg_total != 0 else 0
    cg_acceptance = round((cg_count / cg_total) * 100, 2) if cg_total != 0 else 0
    uplift = tg_acceptance - cg_acceptance
    relative_uplift = (uplift / cg_acceptance) * 100 if cg_acceptance != 0 else 0
    return relative_uplift


st.markdown(f"# Cross-KPI results")


# Crear un DataFrame vacío con las etiquetas de los KPIs como índices y columnas.
kpi_labels = [f'{kpi}_{val}' for kpi in kpi_columns for val in [0, 1]]
matrix_df = pd.DataFrame(index=kpi_labels, columns=kpi_labels)

# Rellenar la matriz con los uplifts relativos.
for row_label, col_label in itertools.product(kpi_labels, repeat=2):
    kpi1, value1 = row_label.split('_')
    kpi2, value2 = col_label.split('_')
    matrix_df.loc[row_label, col_label] = calculate_relative_uplift(dataset, kpi1, kpi2, int(value1), int(value2))

# Mostrar la matriz en la interfaz de usuario
st.write(matrix_df)


# Definir colores para el mapa de calor: verde para valores positivos, rojo para valores negativos.
colors = ['red', 'lightgray', 'green']

# Primero convertimos los valores a numéricos, forzando los no numéricos a NaN
values = pd.to_numeric(matrix_df.values.flatten(), errors='coerce').reshape(matrix_df.values.shape)

# Ahora reemplazamos los NaN por 0
values = np.where(np.isnan(values), 0, values)

# Convierte los valores numéricos a strings con dos decimales
text_values = np.round(values, 0).astype(int).astype(str)
# Concatena '%' a cada elemento individualmente
text = [f"{val}%" for val in text_values.flatten()]
# Vuelve a dar forma al array
text = np.array(text).reshape(values.shape)

fig = go.Figure(data=go.Heatmap(
    z=values,
    x=matrix_df.columns,
    y=matrix_df.index,
    colorscale=colors,
    zmid=0,
    text=text,
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps = False
))


# Ajustar el layout de la figura.
fig.update_layout(
    title='Heatmap of Relative Uplift',
    xaxis_nticks=len(matrix_df.columns),
    yaxis_nticks=len(matrix_df.index)
)

# Mostrar la figura en la interfaz de usuario de Streamlit.
st.plotly_chart(fig)

st.write("Pending to include: 1) Order of column/rows in heatmap,")
st.write("2) Cambiar/resaltar bordes de las cells si significant")
st.write(" 3) related to 2, modify function to support p-value (or adapt calculate_metrics with optional input parameters")
st.write("4) Add filter feature to select certain values of segmentation columns")