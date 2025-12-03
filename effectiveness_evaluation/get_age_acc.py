import pandas as pd


def age_groups_acc_data(file_path,sheet_name,age_num,answer_num):
    """
    Load patient age and model correctness information a given Excel sheet,
    categorize cases into age groups, and compute accuracy statistics.

    This function:
        1. Converts age values to numeric form and handles missing ages as "N/A"
        2. Creates age ranges in decades (e.g., 10–19, 20–29, ..., 80+)
        3. Calculates model accuracy per age group
        4. Counts the number of cases and correct predictions in each group

    Parameters
    ----------
    file_path : str
        Path to the Excel file that contains the age and correctness labels.
    sheet_name : int
        Sheet index in the Excel file to be processed.
    age_num : int
        Column index of the age label for each case in the sheet.
    answer_num : int
        Column index of the correctness indicator (0 = incorrect, 1 = correct).

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame including:
            - age_group : Age bucket (e.g., "30-39", "80+", "N/A")
            - accuracy : Average correctness score per age group
            - case_count : Total number of cases for each sex group
            - right_case_count : Number of correctly diagnosed cases (accuracy == 1)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['age'] = df.iloc[:, age_num]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].apply(
        lambda x: "unclear" if pd.isna(x)
        else int(x)
    )
    df.iloc[:, answer_num] = pd.to_numeric(df.iloc[:, answer_num], errors="coerce")
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df["age_group"] = df["age"].apply(
        lambda x: "N/A" if x == "unclear"
        else "80+" if x >= 80
        else f"{(x // 10) * 10}-{(x // 10) * 10 + 9}")
    accuracy_by_age = df.groupby("age_group")["accuracy"].mean().reset_index()
    case_count = df.groupby("age_group").size().reset_index(name="case_count")
    right_case_count = df.groupby('age_group')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = accuracy_by_age.merge(case_count, on="age_group") \
        .merge(right_case_count, on="age_group")
    result = result[["age_group", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["age_group", "accuracy", "case_count", "right_case_count"]
    print(result)
    return result

def age_groups_acc_all_data(file_path,age_num1,age_num2,age_num3,answer_num1,answer_num2,answer_num3):
    """
    Compute accuracy statistics based on patient age labels from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,age_num,answer_num in zip([df1,df2,df3],[age_num1,age_num2,age_num3],[answer_num1,answer_num2,answer_num3]):
        df['age'] = df.iloc[:, age_num]
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].apply(
            lambda x: "unclear" if pd.isna(x)
            else int(x)
        )
        df.iloc[:, answer_num] = pd.to_numeric(df.iloc[:, answer_num], errors="coerce")
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df = pd.concat(
        [df1[['age', 'accuracy']], df2[['age', 'accuracy']], df3[['age', 'accuracy']]],
        ignore_index=True)
    df["age_group"] = df["age"].apply(
        lambda x: "N/A" if x=="unclear"
        else "80+" if x >= 80
        else f"{(x // 10) * 10}-{(x // 10) * 10 + 9}")
    accuracy_by_age = df.groupby("age_group")["accuracy"].mean().reset_index()
    case_count = df.groupby("age_group").size().reset_index(name="case_count")
    right_case_count = df.groupby('age_group')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = accuracy_by_age.merge(case_count, on="age_group") \
        .merge(right_case_count, on="age_group")
    result = result[["age_group", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["age_group", "accuracy", "case_count", "right_case_count"]
    print(result)
    return result

def main():
    Lancet_result = age_groups_acc_data('label summary.xlsx',0, 5,12)
    NEJM_result = age_groups_acc_data('label summary.xlsx',1, 5,12)
    JAMA_result = age_groups_acc_data('label summary.xlsx',2, 5,12)
    all_result = age_groups_acc_all_data('label summary.xlsx',5, 5,5,12,12,12)

if __name__ == "__main__":
    main()