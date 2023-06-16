import pandas as pd
import numpy as np
import math
import scipy.special as scsp

significance_treshold = 0.1


def z2p(z):
    """From z-score return p-value."""
    return 2*(1- (0.5 * (1 + scsp.erf(abs(z) / math.sqrt(2)))))

def zscore(p1, p2, n1, n2): # p1, p2 proportions
	p = (p1*float(n1) + p2*float(n2))/(float(n1) + float(n2))
	numerator = p1 - p2
	denominator = math.sqrt(p*(1-p)*((1/n1)+(1/n2)))
	return numerator/denominator

def kadane_algorithm(input_list):
    max_current = max_global = input_list[0]
    start = end = 0
    for i in range(1, len(input_list)):
        if input_list[i] > max_current + input_list[i]:
            max_current = input_list[i]
            start = i
        else:
            max_current += input_list[i]
        if max_current > max_global:
            max_global = max_current
            end = i
    return max_global, start, end



def infer_datatypes_and_metatypes(dataset: pd.DataFrame) -> pd.DataFrame:
    info_data = {'COLUMN': [], 'DATATYPE': [], 'METATYPE': []}

    for column in dataset.columns:
        dtype = dataset[column].dtype

        if column == 'CUSTOMERNUMBER':
            info_data['COLUMN'].append(column)
            info_data['DATATYPE'].append('NUMERIC')
            info_data['METATYPE'].append('PK')
        elif column == 'TGCG':
            info_data['COLUMN'].append(column)
            info_data['DATATYPE'].append('STRING')
            info_data['METATYPE'].append('TGCG')
        else:
            info_data['COLUMN'].append(column)

            if dtype == 'bool':
                info_data['DATATYPE'].append('BOOL')
                info_data['METATYPE'].append('KPI')
            elif dtype == 'object':
                info_data['DATATYPE'].append('STRING')
                info_data['METATYPE'].append('SF')
            elif dataset[column].nunique() < 10:
                info_data['DATATYPE'].append('NUM_ST')
                info_data['METATYPE'].append('SF')
            else:
                info_data['DATATYPE'].append('NUMERIC')
                info_data['METATYPE'].append('SF')

    inferred_info_dataset = pd.DataFrame(info_data)
    return inferred_info_dataset

def validate_datatypes_and_metatypes(dataset: pd.DataFrame, info_dataset: pd.DataFrame) -> bool:
    datatype_values = ['BOOL', 'STRING', 'NUM_ST', 'NUMERIC']
    metatype_values = ['TGCG', 'PK', 'KPI', 'SF']

    for index, row in info_dataset.iterrows():
        column = row['COLUMN']
        datatype = row['DATATYPE']
        metatype = row['METATYPE']

        if datatype not in datatype_values or metatype not in metatype_values:
            return False

        if metatype == 'TGCG' and not dataset[column].apply(lambda x: x.lower() in ['target', 'control']).all():
            return False

        if datatype == 'BOOL' and not dataset[column].apply(lambda x: isinstance(x, bool)).all():
            return False

        if datatype == 'STRING' and not dataset[column].apply(lambda x: isinstance(x, str)).all():
            return False

        if datatype == 'NUM_ST' and not (dataset[column].nunique() < 10 and dataset[column].apply(lambda x: isinstance(x, (int, float))).all()):
            return False

        if datatype == 'NUMERIC' and not dataset[column].apply(lambda x: isinstance(x, (int, float))).all():
            return False

        if metatype == 'KPI' and not dataset[column].apply(lambda x: isinstance(x, (int, float))).all():
            return False

    return True

def format_float(value):
    if isinstance(value, float):
        return "{:.2f}".format(value)
    return value


def calculate_metrics2(subset, kpi, tgcg_column):
    """Calculates metrics for a specific KPI."""
    metrics = []
    tg_acceptors = subset.loc[subset[tgcg_column] == 'target', kpi].sum()
    tg_total = len(subset.loc[subset[tgcg_column] == 'target'])
    tg_acceptance = round((tg_acceptors / tg_total)*100,2) if tg_total != 0 else 0

    cg_acceptors = subset.loc[subset[tgcg_column] == 'control', kpi].sum()
    cg_total = len(subset.loc[subset[tgcg_column] == 'control'])
    cg_acceptance = round((cg_acceptors / cg_total) * 100, 2) if cg_total != 0 else 0

    uplift = tg_acceptance - cg_acceptance
    p_value = z2p(zscore(float(tg_acceptors)/float(tg_total), float(cg_acceptors)/float(cg_total),float(tg_total), float(cg_total))) if tg_total != 0 and cg_total != 0 else None
    
    metrics.append([kpi, "{:.2f}".format(tg_acceptors), "{:.2f}".format(tg_acceptance), "{:.2f}".format(cg_acceptors), "{:.2f}".format(cg_acceptance), "{:.2f}".format(uplift), p_value])
    result_df = pd.DataFrame(metrics, columns=["KPI", "TG Acceptors", "TG Acceptance (%)", "CG Acceptors", "CG Acceptance (%)", "Uplift (%)", "P-value"])
    result_df['P-value'] = pd.to_numeric(result_df['P-value'], errors='coerce')
    return result_df


def highlight_pvalue(row):
    """Highlights rows with P-value <= significance_treshold."""
    if float(row["P-value"]) <= significance_treshold:
        return ["background-color: lightgreen"] * len(row)
    return [""] * len(row)