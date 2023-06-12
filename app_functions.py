import pandas as pd
import numpy as np
import math
import scipy.special as scsp

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