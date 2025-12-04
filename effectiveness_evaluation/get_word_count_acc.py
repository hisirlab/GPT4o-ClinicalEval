import json
import pandas as pd

def word_count_acc_data(excel_file, sheet_name, word_count_num,answer_num):
    """
    Analyze the relationship between word count and model accuracy within a single dataset.

    This function processes a single Excel sheet to:
    1. Load word count and accuracy data
    2. Group word counts into 50-word bins
    3. Calculate aggregated statistics for each word count group

    Word Count Grouping Strategy
    ----------------------------
    • 50-word bins: Cases are grouped into ranges of 50 words
    • Special handling: All cases with ≥400 words are grouped as "400+"
    • Group labels: "0-49", "50-99", ..., "350-399", "400+"

    Statistical Aggregation
    -----------------------
    For each word count group, the function calculates:
    • Average actual word count (mean of raw word counts)
    • Mean accuracy across cases in the group
    • Total number of cases in the group

    Parameters
    ----------
    excel_file : str
        Path to the Excel file containing the dataset
    sheet_name : int
        Index of the Excel sheet to process
    word_count_num : int
        Column index containing word count values
    answer_num : int
        Column index containing correctness labels (0=incorrect, 1=correct)

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame with columns:
            - word_count_group : Word count range category
            - word_count : Average actual word count within the group
            - accuracy : Mean accuracy (0-1) for the group
            - case_count : Number of cases in the group
        Sorted by increasing word count ranges.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1)
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df['word_count'] = df.iloc[:, word_count_num]
    df['word_count_group'] = (df['word_count'] // 50) * 50
    df['word_count_group'] = df['word_count_group'].apply(
        lambda x: "400+" if x >= 400 else f"{(x // 50) * 50}-{(x // 50) * 50 + 49}")
    average_counts = df.groupby('word_count_group')['word_count'].mean().reset_index()
    average_accuracy = df.groupby('word_count_group')['accuracy'].mean().reset_index()
    case_count = df.groupby('word_count_group').size().reset_index(name='case_count')
    result = pd.merge(average_counts, average_accuracy, on="word_count_group")
    result = pd.merge(result, case_count, on="word_count_group")
    result['order'] = result['word_count_group'].apply(lambda x: 9999 if x == "400+" else int(x.split('-')[0]))
    result = result.sort_values('order').drop(columns='order').reset_index(drop=True)
    print(result)
    return result

def word_count_acc_all_data(excel_file, sheet_name1, sheet_name2, sheet_name3,word_count_num1,word_count_num2,word_count_num3,answer_num1, answer_num2, answer_num3):
    """
    Compute accuracy statistics based on word count labels from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    sheet_names = [sheet_name1, sheet_name2, sheet_name3]
    dfs = [pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1) for sheet_name in sheet_names]
    df_excel1, df_excel2, df_excel3 = dfs
    dfs = []
    for df, word_count_num, answer_num in zip([df_excel1, df_excel2, df_excel3],[word_count_num1,word_count_num2,word_count_num3],[answer_num1, answer_num2, answer_num3]):
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
        df['word_count'] = df.iloc[:, word_count_num]
        dfs.append(df)
    df = pd.concat([df[["word_count", "accuracy"]] for df in dfs], ignore_index=True)
    df['word_count_group'] = (df['word_count'] // 50) * 50
    df['word_count_group'] = df['word_count_group'].apply(
        lambda x: "400+" if x >= 400 else f"{(x // 50) * 50}-{(x // 50) * 50 + 49}")
    average_counts = df.groupby('word_count_group')['word_count'].mean().reset_index()
    average_accuracy = df.groupby('word_count_group')['accuracy'].mean().reset_index()
    case_count = df.groupby('word_count_group').size().reset_index(name='case_count')
    result = pd.merge(average_counts, average_accuracy, on="word_count_group")
    result = pd.merge(result, case_count, on="word_count_group")
    result['order'] = result['word_count_group'].apply(lambda x: 9999 if x == "400+" else int(x.split('-')[0]))
    result = result.sort_values('order').drop(columns='order').reset_index(drop=True)
    print(result)
    return result

def main():
    Lancet_result = word_count_acc_data('label summary.xlsx', 0, 13,14)
    NEJM_result = word_count_acc_data('label summary.xlsx',1,13,14)
    JAMA_result = word_count_acc_data('label summary.xlsx', 2, 13,14)
    all_result= word_count_acc_all_data('label summary.xlsx',
                                        0,1,2,
                                        13,13,13,
                                        14,14,14)

if __name__ == "__main__":
    main()

