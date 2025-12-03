import pandas as pd

def image_count_acc_data(file_path,sheet_name,image_count_num,answer_num):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['image_count'] = df.iloc[:, image_count_num]
    df["image_count"] = pd.to_numeric(df["image_count"], errors="coerce")
    df = df.dropna(subset=["image_count"])
    df["image_count"] = df["image_count"].astype(int)
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    accuracy_by_image_count = df.groupby("image_count")["accuracy"].mean().reset_index()
    case_count = df.groupby('image_count').size().reset_index(name='case_count')
    result = pd.merge(accuracy_by_image_count, case_count, on="image_count")
    result = result[["image_count", "accuracy", "case_count"]]
    result.columns = ["image_count", "accuracy", "case_count"]
    print(result)
    return result

def image_count_acc_all_merge_data(file_path,image_count_num1,image_count_num2,image_count_num3,answer_num1,answer_num2,answer_num3):
    """
    Load image count and model correctness data from three Excel sheets,
    categorize cases based on number of images, and compute aggregated statistics.

    • The number of images per case is extracted from each journal dataset.
    • Cases are grouped into:
            - 1, 2, 3, 4 images
            - >=5 images (merged into a single category)
    • Model performance statistics are then merged across all journals.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the three datasets.
    image_count_num1, image_count_num2, image_count_num3 : int
        Column indices for the number of images in each sheet.
    answer_num1, answer_num2, answer_num3 : int
        Column indices for correctness labels (0=wrong, 1=correct)
        in each respective sheet.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing:
            - image_count : Grouped number of images per case (1–4, ">=5")
            - accuracy : Mean accuracy within each group
            - case_count : Number of cases per image count group
            - right_case_count : Number of correctly predicted cases per group
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,image_count_num,answer_num in zip([df1,df2,df3],[image_count_num1,image_count_num2,image_count_num3],[answer_num1,answer_num2,answer_num3]):
        df['image_count'] = df.iloc[:, image_count_num]
        df["image_count"] = pd.to_numeric(df["image_count"], errors="coerce")
        df = df.dropna(subset=["image_count"])
        df["image_count"] = df["image_count"].astype(int)
        df["image_count"] = df["image_count"].apply(lambda x: x if x < 5 else ">=5")
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df = pd.concat(
        [df1[['image_count', 'accuracy']], df2[['image_count', 'accuracy']], df3[['image_count', 'accuracy']]],
        ignore_index=True)
    accuracy_by_image_count = df.groupby("image_count")["accuracy"].mean().reset_index()
    case_count = df.groupby('image_count').size().reset_index(name='case_count')
    result = pd.merge(accuracy_by_image_count, case_count, on="image_count")
    right_case_count = df.groupby('image_count')["accuracy"].sum().reset_index(name="right_case_count")
    result = pd.merge(result, right_case_count)
    result = result[["image_count", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["image_count", "accuracy", "case_count", "right_case_count"]
    print(result)
    return result

def main():
    Lancet_result = image_count_acc_data('label summary.xlsx',0,3,12)
    NEJM_result = image_count_acc_data('label summary.xlsx', 1, 3, 12)
    JAMA_result = image_count_acc_data('label summary.xlsx', 2, 3, 12)
    All_result = image_count_acc_all_merge_data('label summary.xlsx', 3, 3, 3,12,12,12)

if __name__ == "__main__":
    main()