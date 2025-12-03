import pandas as pd


def task_type_acc_data(file_path,sheet_name,diagnostic_num,answer_num):
    """
    Compute accuracy statistics based on patient task type labels from a given Excel sheet.

    Parameters
    ----------
    file_path : str
        Path to the Excel file that contains the sex and correctness labels.
    sheet_name : int
        Sheet index in the Excel file to be processed.
    diagnostic_num : int
        Column index of the task type label for each case in the sheet.
    answer_num : int
        Column index of the correctness indicator (0 = incorrect, 1 = correct).

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing accuracy analysis by sex group, including:
            - diagnostic : task type label
            - accuracy : Mean accuracy for each task type group
            - case_count : Total number of cases for each task type group
            - right_case_count : Number of correctly diagnosed cases (accuracy == 1)
            - wrong_case_count : Number of incorrectly diagnosed cases (accuracy == 0)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['diagnostic'] = df.iloc[:, diagnostic_num]
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    accuracy_by_diagnostic = df.groupby("diagnostic")["accuracy"].mean().reset_index()
    case_count = df.groupby('diagnostic').size().reset_index(name='case_count')
    right_case_count = df.groupby('diagnostic')["accuracy"].sum().reset_index(name="right_case_count")
    wrong_case_count = (df.groupby('diagnostic')["accuracy"].count() - right_case_count).reset_index(name="wrong_case_count")
    result = accuracy_by_diagnostic.merge(case_count, on="diagnostic") \
        .merge(right_case_count, on="diagnostic") \
        .merge(wrong_case_count, on="diagnostic")
    diagnostic_mapping = {
        0: "Diagnosis", 1: "Non-diagnosis"
    }
    result["diagnostic"] = result["diagnostic"].replace(diagnostic_mapping)
    result = result[["diagnostic", "accuracy", "case_count",'right_case_count','wrong_case_count']]
    result.columns = ["diagnostic", "accuracy", "case_count",'right_case_count','wrong_case_count']
    print(result)
    return result


def task_type_acc_all_data(file_path,diagnostic_num1,diagnostic_num2,diagnostic_num3,answer_num1,answer_num2,answer_num3):
    """
    Compute accuracy statistics based on task type labels from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,diagnostic_num,answer_num in zip([df1,df2,df3],[diagnostic_num1,diagnostic_num2,diagnostic_num3],[answer_num1,answer_num2,answer_num3]):
        df['diagnostic'] = df.iloc[:, diagnostic_num]
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df = pd.concat(
        [df1[['diagnostic', 'accuracy']], df2[['diagnostic', 'accuracy']], df3[['diagnostic', 'accuracy']]],
        ignore_index=True)
    accuracy_by_sex = df.groupby("diagnostic")["accuracy"].mean().reset_index()
    case_count = df.groupby('diagnostic').size().reset_index(name='case_count')
    right_case_count = df.groupby('diagnostic')["accuracy"].sum().reset_index(name="right_case_count")
    wrong_case_count = (df.groupby('diagnostic')["accuracy"].count() - right_case_count).reset_index(name="wrong_case_count")
    result = accuracy_by_sex.merge(case_count, on="diagnostic") \
        .merge(right_case_count, on="diagnostic") \
        .merge(wrong_case_count, on="diagnostic")
    diagnostic_mapping = {
        0: "Diagnosis", 1: "Non-diagnosis"
    }
    result["diagnostic"] = result["diagnostic"].replace(diagnostic_mapping)
    result = result[["diagnostic", "accuracy", "case_count", 'right_case_count', 'wrong_case_count']]
    result.columns = ["diagnostic", "accuracy", "case_count", 'right_case_count', 'wrong_case_count']
    print(result)
    return result

def main():
    Lancet_result = task_type_acc_data('label summary.xlsx',0,7,12)
    NEJM_result = task_type_acc_data('label summary.xlsx',1,7,12)
    JAMA_result = task_type_acc_data('label summary.xlsx', 1, 7, 12)
    All_result = task_type_acc_all_data('label summary.xlsx',7,7,7,12,12,12)

if __name__ == "__main__":
    main()
