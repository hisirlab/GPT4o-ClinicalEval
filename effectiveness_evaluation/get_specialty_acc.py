import json
import re

import pandas as pd

def specialty_acc_data(file_path,sheet_name,specialty_num,answer_num):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['specialty'] = df.iloc[:, specialty_num]
    df["specialty"] = df["specialty"].astype(str).fillna("").apply(lambda x: x if "," not in x else "/")
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    accuracy_by_specialty = df.groupby("specialty")["accuracy"].mean().reset_index()
    case_count = df.groupby('specialty').size().reset_index(name='case_count')
    right_case_count = df.groupby('specialty')["accuracy"].sum().reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_specialty, case_count, on="specialty")
    result = pd.merge(result, right_case_count, on="specialty")
    result = result[["specialty", "accuracy", "case_count","right_case_count"]]
    result.columns = ["specialty", "accuracy", "case_count","right_case_count"]
    specialty_mapping = {
        '0': "Others", '1': "Orthopedics", '2': "Plastic Surgery", '3': "Cardiology", '4': "Urology",
        '5': "Gastroenterology", '6': "Radiology",
        '7': "Dermatology", '8': "Anesthesiology", '9': "Oncology", '10': "Otolaryngology", '11': "General Surgery",
        '12': "Ophthalmology", '13': "Critical Care", '14': "Pulmonary Medicine", '15': "Emergency Medicine",
        '16': "Pathology", '17': "Ob/Gyn", '18': "Neurology", '19': "Nephrology",
        '20': "Physical Medicine & Rehabilitation", '21': "Psychiatry",
        '22': "Allergy & Immunology", '23': "Rheumatology", '24': "Internal Medicine", '25': "Family Medicine",
        '26': "Public Health & Preventive Medicine",
        '27': "Infectious Diseases", '28': "Pediatrics", '29': "Diabetes & Endocrinology", '30': "Hematology",
        '31': "Stomatology", '32': "Cardio-Thoracic Surgery", '33': "Neurosurgery"
    }
    result["specialty"] = result["specialty"].replace(specialty_mapping)
    small_cases = result[result['case_count'] < 0]
    large_cases = result[result['case_count'] >= 0]
    if not small_cases.empty:
        other_row = pd.DataFrame({
            "specialty": ["Others"],
            "accuracy": [small_cases["right_case_count"].sum() / small_cases["case_count"].sum()],
            "case_count": [small_cases["case_count"].sum()],
            "right_case_count": [small_cases["right_case_count"].sum()]
        })
        result = pd.concat([large_cases, other_row], ignore_index=True)
    result = result.sort_values(by=['accuracy', 'case_count'], ascending=[False, False]).reset_index(drop=True)
    print(result)
    return result

def specialty_acc_all_data(file_path,specialty_num1,specialty_num2,specialty_num3,answer_num1,answer_num2,answer_num3):
    """
    Load specialty and model correctness data from three Excel sheets,
    map specialty codes to specialties, and compute performance statistics.

    • Specialty codes are extracted from three journal datasets and converted into category names.
    • Cases from all sheets are merged together.
    • Accuracy per specialty category is computed.
    • Specialty categories with small sample sizes (case_count < 11) are merged into “Others” to ensure statistical reliability.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the three specialty-coded datasets.
    specialty_num1, specialty_num2, specialty_num3 : int
        Column indices for specialty codes in each sheet.
    answer_num1, answer_num2, answer_num3 : int
        Column indices of correctness labels (0=wrong, 1=correct)
        in each respective sheet.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing aggregated specialty performance statistics:
            - specialty : Specialty (mapped from specialty numeric code)
            - accuracy : Mean accuracy within the category
            - case_count : Number of cases per category
            - right_case_count : Count of correctly predicted cases
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,specialty_num,answer_num in zip([df1,df2,df3],[specialty_num1,specialty_num2,specialty_num3],[answer_num1,answer_num2,answer_num3]):
        df['specialty'] = df.iloc[:, specialty_num]
        df["specialty"] = df["specialty"].astype(str).fillna("").apply(lambda x: x if "," not in x else "/")
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df = pd.concat([df1[['specialty', 'accuracy']], df2[['specialty', 'accuracy']], df3[['specialty', 'accuracy']]],ignore_index=True)
    accuracy_by_specialty = df.groupby("specialty")["accuracy"].mean().reset_index()
    case_count = df.groupby('specialty').size().reset_index(name='case_count')
    right_case_count = df.groupby('specialty')["accuracy"].sum().reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_specialty, case_count, on="specialty")
    result = pd.merge(result, right_case_count, on="specialty")
    result = result[["specialty", "accuracy", "case_count","right_case_count"]]
    result.columns = ["specialty", "accuracy", "case_count","right_case_count"]
    specialty_mapping = {
        '0': "Others", '1': "Orthopedics", '2': "Plastic Surgery", '3': "Cardiology", '4': "Urology", '5': "Gastroenterology", '6': "Radiology",
        '7': "Dermatology", '8': "Anesthesiology", '9': "Oncology", '10': "Otolaryngology",'11': "General Surgery",
        '12': "Ophthalmology", '13': "Critical Care", '14': "Pulmonary Medicine", '15': "Emergency Medicine",
        '16': "Pathology", '17': "Ob/Gyn",'18': "Neurology", '19': "Nephrology", '20': "Physical Medicine & Rehabilitation", '21': "Psychiatry",
        '22': "Allergy & Immunology", '23': "Rheumatology", '24': "Internal Medicine", '25': "Family Medicine", '26': "Public Health & Preventive Medicine",
        '27': "Infectious Diseases", '28': "Pediatrics",'29': "Diabetes & Endocrinology", '30': "Hematology",
        '31': "Stomatology", '32': "Cardio-Thoracic Surgery", '33': "Neurosurgery", "/": "Others"
    }
    result["specialty"] = result["specialty"].replace(specialty_mapping)
    small_cases = result[result['case_count'] < 11]
    large_cases = result[result['case_count'] >= 11]
    if not small_cases.empty:
        other_row = pd.DataFrame({
            "specialty": ["Others"],
            "accuracy": [small_cases["right_case_count"].sum() / small_cases["case_count"].sum()],
            "case_count": [small_cases["case_count"].sum()],
            "right_case_count": [small_cases["right_case_count"].sum()]
        })
        result = pd.concat([large_cases, other_row], ignore_index=True)
    result = result[result["specialty"] != "Others"]
    result = result.sort_values(by=['accuracy', 'case_count'], ascending=[False, False]).reset_index(drop=True)
    print(result)
    return result

def main():
    Lancet_result = specialty_acc_data('label summary.xlsx',0,9,12)
    NEJM_result = specialty_acc_data('label summary.xlsx', 1, 9, 12)
    JAMA_result = specialty_acc_data('label summary.xlsx', 2, 9, 12)
    All_result = specialty_acc_all_data('label summary.xlsx',9,9,9,12,12,12)

if __name__ == "__main__":
    main()