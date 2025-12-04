import json
import pandas as pd

def word_count_acc_data(json_file, excel_file, sheet_name, answer_num):
    """
    Analyze the relationship between textual word count and model accuracy by
    merging JSON text data with Excel evaluation results and grouping by word count ranges.

    1. Data Loading:
        - Load text data from JSON file containing case text inputs
        - Load accuracy labels from Excel evaluation results
    2. Data Merging: Combine text data with accuracy labels using case IDs
    3. Word Count Calculation: Count words in each text input
    4. Grouping: Categorize cases into 50-word bins (0-49, 50-99, ..., 350-399, 400+)
    5. Aggregation: Calculate average word count and accuracy for each word count group
    6. Sorting: Order groups by word count range for logical presentation

    Grouping Strategy
    -----------------
    • 50-word bins: Each group spans 50 words (e.g., 0-49, 50-99)
    • Special handling for long texts: All texts with ≥400 words grouped as "400+"
    • Group labels: "0-49", "50-99", ..., "350-399", "400+"

    Parameters
    ----------
    json_file : str
        Path to JSON file containing text input data with fields:
        - "ID": Unique case identifier (string)
        - "text_input": Text content for word count analysis
    excel_file : str
        Path to Excel file containing accuracy evaluation results
    sheet_name : int
        Index of the Excel sheet containing accuracy data
    answer_num : int
            Column index in the Excel sheet containing correctness labels (0=incorrect, 1=correct)

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing:
            - word_count_group : Word count range category (e.g., "0-49", "400+")
            - word_count : Average actual word count within the group
            - accuracy : Mean accuracy (0-1) for cases in the group
            - case_count : Number of cases in the word count group
        Sorted by increasing word count ranges.
    """
    with open(json_file, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    df_json = pd.DataFrame(json_data)
    df_excel = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1)
    df_json["case_id"] = df_json["ID"].astype(str)
    df_excel['case_id'] = df_excel.iloc[:, 0].astype(str)
    df_excel.iloc[:, answer_num] = pd.to_numeric(df_excel.iloc[:, answer_num], errors="coerce")
    df_excel['accuracy'] = df_excel.iloc[:, answer_num].astype(int)
    df = df_json.merge(df_excel, on="case_id", how="left")
    df["text_input"] = df["text_input"].fillna("").astype(str)
    df['word_count'] = df["text_input"].str.split().str.len()
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

def word_count_acc_all_data(json_file1, json_file2, json_file3, excel_file, sheet_name1, sheet_name2, sheet_name3,answer_num1, answer_num2, answer_num3):
    """
    Compute accuracy statistics based on word count labels from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    json_files = [json_file1, json_file2, json_file3]
    dfs = [pd.DataFrame(json.load(open(f, 'r', encoding='utf-8'))) for f in json_files]
    df_json1, df_json2, df_json3 = dfs
    sheet_names = [sheet_name1, sheet_name2, sheet_name3]
    dfs = [pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1) for sheet_name in sheet_names]
    df_excel1, df_excel2, df_excel3 = dfs
    dfs = []
    for df_json, df_excel, answer_num in zip([df_json1, df_json2, df_json3], [df_excel1, df_excel2, df_excel3],[answer_num1, answer_num2, answer_num3]):
        df_json["case_id"] = df_json["ID"].astype(str)
        df_excel['case_id'] = df_excel.iloc[:, 0].astype(str)
        df_excel.iloc[:, answer_num] = pd.to_numeric(df_excel.iloc[:, answer_num], errors="coerce")
        df_excel['accuracy'] = df_excel.iloc[:, answer_num].astype(int)
        merged_df = df_json.merge(df_excel, on="case_id", how="left")
        dfs.append(merged_df)
    df = pd.concat([df[["text_input", "accuracy"]] for df in dfs], ignore_index=True)
    df["text_input"] = df["text_input"].fillna("").astype(str)
    df['word_count'] = df["text_input"].str.split().str.len()
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
    Lancet_result = word_count_acc_data('Lancet_json_record.json', 'label summary.xlsx', 0, 12)
    NEJM_result = word_count_acc_data('NEJM_json_record.json','label summary.xlsx',1,12)
    JAMA_result = word_count_acc_data('JAMA_json_record.json', 'label summary.xlsx', 2, 12)
    all_result= word_count_acc_all_data('Lancet_json_record.json','NEJM_json_record.json','JAMA_json_record.json',
                                        'label summary.xlsx',
                                        0,1,2,
                                        12,12,12)

if __name__ == "__main__":
    main()

