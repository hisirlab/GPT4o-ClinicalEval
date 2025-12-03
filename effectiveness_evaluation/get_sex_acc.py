import pandas as pd

def sex_acc_data(file_path,sheet_name,sex_num,answer_num):
    """
    Compute accuracy statistics based on patient sex labels from a given Excel sheet.

    Parameters
    ----------
    file_path : str
        Path to the Excel file that contains the sex and correctness labels.
    sheet_name : int
        Sheet index in the Excel file to be processed.
    sex_num : int
        Column index of the sex label for each case in the sheet.
    answer_num : int
        Column index of the correctness indicator (0 = incorrect, 1 = correct).

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing accuracy analysis by sex group, including:
            - sex : Sex category label
            - accuracy : Mean accuracy for each sex group
            - case_count : Total number of cases for each sex group
            - right_case_count : Number of correctly diagnosed cases (accuracy == 1)
            - wrong_case_count : Number of incorrectly diagnosed cases (accuracy == 0)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['sex'] = df.iloc[:, sex_num]
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    accuracy_by_sex = df.groupby("sex")["accuracy"].mean().reset_index()
    case_count = df.groupby('sex').size().reset_index(name='case_count')
    right_case_count = df.groupby('sex')["accuracy"].sum().reset_index(name="right_case_count")
    wrong_case_count = (df.groupby('sex')["accuracy"].count() - right_case_count).reset_index(name="wrong_case_count")
    result = accuracy_by_sex.merge(case_count, on="sex") \
        .merge(right_case_count, on="sex") \
        .merge(wrong_case_count, on="sex")
    result = result[["sex", "accuracy", "case_count",'right_case_count','wrong_case_count']]
    result.columns = ["sex", "accuracy", "case_count",'right_case_count','wrong_case_count']
    print(result)
    return result

def sex_acc_data_all(file_path,sex_num1,sex_num2,sex_num3,answer_num1,answer_num2,answer_num3):
    """
    Compute accuracy statistics based on patient sex labels from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    df1['sex'] = df1.iloc[:, sex_num1]
    df2['sex'] = df2.iloc[:, sex_num2]
    df3['sex'] = df3.iloc[:, sex_num3]
    df1['accuracy'] = df1.iloc[:, answer_num1].astype(int)
    df2['accuracy'] = df2.iloc[:, answer_num2].astype(int)
    df3['accuracy'] = df3.iloc[:, answer_num3].astype(int)
    df = pd.concat(
        [df1[['sex', 'accuracy']], df2[['sex', 'accuracy']], df3[['sex', 'accuracy']]],
        ignore_index=True)
    accuracy_by_sex = df.groupby("sex")["accuracy"].mean().reset_index()
    case_count = df.groupby('sex').size().reset_index(name='case_count')
    right_case_count = df.groupby('sex')["accuracy"].sum().reset_index(name="right_case_count")
    wrong_case_count = (df.groupby('sex')["accuracy"].count() - right_case_count).reset_index(name="wrong_case_count")
    result = accuracy_by_sex.merge(case_count, on="sex") \
        .merge(right_case_count, on="sex") \
        .merge(wrong_case_count, on="sex")
    result = result[["sex", "accuracy", "case_count", 'right_case_count', 'wrong_case_count']]
    result.columns = ["sex", "accuracy", "case_count", 'right_case_count', 'wrong_case_count']
    print(result)
    return result

def main():
    Lancet_result=sex_acc_data('label summary.xlsx',0, 6,12)
    NEJM_result=sex_acc_data('label summary.xlsx',0, 6,12)
    JAMA_result=sex_acc_data('label summary.xlsx',0, 6,12)
    All_result=sex_acc_data_all('label summary.xlsx',6,6,6,12,12,12)

if __name__ == "__main__":
    main()

