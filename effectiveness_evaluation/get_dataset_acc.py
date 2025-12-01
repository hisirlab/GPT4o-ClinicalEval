import pandas as pd

def dataset_acc_data(excel_file,date_num,answer_num):
    """
    Compute accuracy statistics for three journals (Lancet, NEJM, JAMA)
    based on an Excel file that stores:
        - the date of each case
        - whether the model answered the case correctly (0/1)

    This function:
    1. Reads three sheets (sheet 0/1/2) from an Excel file, each corresponding to one dataset.
    2. Extracts the date column and accuracy column by index.
    3. Converts accuracy values to integer (0/1).
    4. Splits data into:
        - Full dataset
        - Post-knowledge-cutoff subset
    5. Computes, for each journal and each split:
        - mean accuracy
        - number of cases
        - correctly answered cases
        - incorrectly answered cases
    6. Aggregates results across all journals ("All").


    Parameters
    ----------
    excel_file : str
        Path to the Excel file containing the date and correctness labels for each case.
    date_num : int
        Column index of the date field in each sheet.
    answer_num : int
        Column index of the model correctness indicator (0=wrong, 1=correct).


    Returns
    -------
    final_result : pandas.DataFrame
        A DataFrame summarizing accuracy statistics with columns:
            - dataset       : 'Lancet', 'NEJM', 'JAMA', or 'All'
            - post          : 'Full' or 'Post'
            - accuracy      : mean accuracy
            - case_count    : number of cases
            - right_case_count
            - wrong_case_count
    """
    final_result=pd.DataFrame()
    right_case_count_sum_full = 0
    wrong_case_count_sum_full = 0
    right_case_count_sum_post = 0
    wrong_case_count_sum_post = 0

    for sheet_name,dataset in zip([0,1,2],['Lancet','NEJM','JAMA']):
        df_full = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1)
        df_full["date"] = pd.to_datetime(df_full.iloc[:, date_num], errors='coerce')
        knowledge_cutoff_date = pd.to_datetime("2023-10-01")
        df_full['accuracy'] = pd.to_numeric(df_full.iloc[:, answer_num], errors="coerce").astype(int)
        df_post = df_full[df_full["date"] > knowledge_cutoff_date]
        for post,df in zip(["Full","Post"],[df_full,df_post]):
            right_case_count = (df["accuracy"] == 1).sum()
            wrong_case_count = (df["accuracy"] == 0).sum()
            accuracy_value = df["accuracy"].mean()
            if post=="Full":
                right_case_count_sum_full += right_case_count
                wrong_case_count_sum_full += wrong_case_count
            else:
                right_case_count_sum_post += right_case_count
                wrong_case_count_sum_post += wrong_case_count

            result = pd.DataFrame({
                "dataset": [dataset],
                "post": [post],
                "accuracy": [accuracy_value],
                "case_count": [right_case_count + wrong_case_count],
                "right_case_count": [right_case_count],
                "wrong_case_count": [wrong_case_count]
            })
            final_result = pd.concat([final_result, result], ignore_index=True)
    for post,right_case_count_sum,wrong_case_count_sum in zip(["Full","Post"],[right_case_count_sum_full,right_case_count_sum_post],[wrong_case_count_sum_full,wrong_case_count_sum_post]):
        result = pd.DataFrame({
            "dataset": "All",
            "post": [post],
            "accuracy": [right_case_count_sum / (right_case_count_sum + wrong_case_count_sum)],
            "case_count": [right_case_count_sum + wrong_case_count_sum],
            "right_case_count": [right_case_count_sum],
            "wrong_case_count": [wrong_case_count_sum]
        })
        final_result = pd.concat([final_result, result], ignore_index=True)
    print(final_result)
    return final_result

def main():
    final_result=dataset_acc_data('label summary.xlsx',12,14)

if __name__ == "__main__":
    main()