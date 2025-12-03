import json
import re
import pandas as pd

def icd_acc_data(file_path,sheet_name,icd_num,answer_num):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['icd'] = df.iloc[:, icd_num]
    df["icd"] = df["icd"].astype(str)
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    accuracy_by_icd = df.groupby("icd")["accuracy"].mean().reset_index()
    case_count = df.groupby('icd').size().reset_index(name='case_count')
    right_case_count = df.groupby('icd')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_icd, case_count, on="icd")
    result = pd.merge(result, right_case_count, on="icd")
    result = result[["icd", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["icd", "accuracy", "case_count", "right_case_count"]
    icd_mapping = {
        '1': "Infectious", '2': "Neoplasms", '3': "Blood", '4': "Immune",
        '5': "Metabolic", '6': "Mental", '7': "Sleep", '8': "Nervous", '9': "Visual",
        '10': "Ear", '11': "Circulatory", '12': "Respiratory", '13': "Digestive", '14': "Skin",
        '15': "Musculoskeletal", '16': "Genitourinary", '17': "Sexual", '18': "Pregnancy",
        '19': "Perinatal", '20': "Developmental", '21': "Symptoms", '22': "Injury",
        '23': "External", '24': "Factors", '25': "Special", '26': "Traditional",
        '27': "Functioning", '28': "Extension", '0':"Others"
    }
    result["icd"] = result["icd"].replace(icd_mapping)
    result = result.sort_values(by=['accuracy', 'case_count'], ascending=[False, False]).reset_index(drop=True)
    print(result)
    return result

def icd_acc_all_data(file_path,icd_num1,icd_num2,icd_num3,answer_num1,answer_num2,answer_num3):
    """
    Load ICD category and model correctness data from three Excel sheets,
    map ICD numeric codes to disease categories, and compute performance statistics.

    • ICD codes are extracted from three journal datasets and converted into category names.
    • Cases from all sheets are merged together.
    • Accuracy per ICD disease category is computed.
    • ICD categories with small sample sizes (case_count < 11) are merged into “Others” to ensure statistical reliability.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the three ICD-coded datasets.
    icd_num1, icd_num2, icd_num3 : int
        Column indices for ICD codes in each sheet.
    answer_num1, answer_num2, answer_num3 : int
        Column indices of correctness labels (0=wrong, 1=correct)
        in each respective sheet.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing aggregated ICD performance statistics:
            - icd : Disease category (mapped from ICD numeric code)
            - accuracy : Mean accuracy within the category
            - case_count : Number of cases per category
            - right_case_count : Count of correctly predicted cases
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,icd_num,answer_num in zip([df1,df2,df3],[icd_num1,icd_num2,icd_num3],[answer_num1,answer_num2,answer_num3]):
        df['icd'] = df.iloc[:, icd_num]
        df["icd"] = df["icd"].astype(str)
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df = pd.concat([df1[['icd', 'accuracy']], df2[['icd', 'accuracy']], df3[['icd', 'accuracy']]],ignore_index=True)
    accuracy_by_icd = df.groupby("icd")["accuracy"].mean().reset_index()
    case_count = df.groupby('icd').size().reset_index(name='case_count')
    right_case_count = df.groupby('icd')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_icd, case_count, on="icd")
    result = pd.merge(result, right_case_count, on="icd")
    result = result[["icd", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["icd", "accuracy", "case_count", "right_case_count"]
    icd_mapping = {
        '1': "Infectious", '2': "Neoplasms", '3': "Blood", '4': "Immune",
        '5': "Metabolic", '6': "Mental", '7': "Sleep", '8': "Nervous", '9': "Visual",
        '10': "Ear", '11': "Circulatory", '12': "Respiratory", '13': "Digestive", '14': "Skin",
        '15': "Musculoskeletal", '16': "Genitourinary", '17': "Sexual", '18': "Pregnancy",
        '19': "Perinatal", '20': "Developmental", '21': "Symptoms", '22': "Injury",
        '23': "External", '24': "Factors", '25': "Special", '26': "Traditional",
        '27': "Functioning", '28': "Extension", '0':"Others"
    }
    result["icd"] = result["icd"].replace(icd_mapping)
    small_cases = result[result['case_count'] < 11]
    large_cases = result[result['case_count'] >= 11]
    if not small_cases.empty:
        other_row = pd.DataFrame({
            "icd": ["Others"],
            "accuracy": [small_cases["right_case_count"].sum() / small_cases["case_count"].sum()],
            "case_count": [small_cases["case_count"].sum()],
            "right_case_count": [small_cases["right_case_count"].sum()]
        })
        result = pd.concat([large_cases, other_row], ignore_index=True)
    result = result[result["icd"] != "Others"]
    result = result.sort_values(by=['accuracy', 'case_count'], ascending=[False, False]).reset_index(drop=True)
    print(result)
    return result

def main():
    Lancet_result = icd_acc_data('label summary.xlsx',0,8,12)
    NEJM_result = icd_acc_data('label summary.xlsx', 1, 8, 12)
    JAMA_result = icd_acc_data('label summary.xlsx', 2, 8, 12)
    All_result = icd_acc_all_data('label summary.xlsx',8,8,8,12,12,12)

if __name__ == "__main__":
    main()