import pandas as pd

def image_pixel_group(df):
    """
    Aggregate image pixel count data into groups and calculate accuracy statistics.

    Processes pixel group data to:
    1. Calculate accuracy metrics for each pixel group
    2. Format group labels for visualization
    3. Handle special cases for high pixel counts (≥30)

    Grouping Strategy:
    • Groups <30: Labeled as "X-(X+2)" (e.g., "0-2", "2-4")
    • Groups ≥30: Labeled as "30+" (combined high pixel count category)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing:
            - pixel_group : Numeric pixel group identifier
            - answer_4o : Binary correctness labels (0=incorrect, 1=correct)

    Returns
    -------
    result_image : pandas.DataFrame
        DataFrame with columns:
            - pixel_group : Formatted group label (e.g., "0-2", "30+")
            - accuracy : Mean accuracy within group
            - case_count : Number of cases in group
            - right_case_count : Number of correctly predicted cases
    """
    accuracy_by_image_token = df.groupby("pixel_group")["answer_4o"].mean().reset_index(name='accuracy')
    case_count = df.groupby('pixel_group').size().reset_index(name='case_count')
    right_case_count = df.groupby('pixel_group')["answer_4o"].sum().reset_index(name="right_case_count")
    result_image = accuracy_by_image_token.merge(case_count, on="pixel_group").merge(right_case_count,on="pixel_group")
    result_image.loc[result_image['pixel_group'] >= 30, 'pixel_group'] = result_image['pixel_group'].astype(str) + "+"
    mask = ~result_image['pixel_group'].astype(str).str.endswith("+")
    result_image.loc[mask, 'pixel_group'] = (
            result_image.loc[mask, 'pixel_group'].astype(str) + "-" +
            (result_image.loc[mask, 'pixel_group'].astype(int) + 2).astype(str)
    )
    result_image = result_image[["pixel_group", "accuracy", "case_count", "right_case_count"]]
    result_image.columns = ["pixel_group", "accuracy", "case_count", "right_case_count"]
    print(result_image)
    return result_image

def get_image_pixel_acc(excel_file,sheet_name,pixel_num,answer_num):
    """
    Load and preprocess image pixel count data from an Excel sheet.

    Processing Steps:
    1. Load data from specified Excel sheet
    2. Extract correctness labels and pixel counts
    3. Normalize pixel counts (divide by 100,000)
    4. Group pixel counts into 2-unit bins (0-2, 2-4, etc.)
    5. Cap maximum group at 30 (all higher values grouped as 30)

    Parameters
    ----------
    excel_file : str
        Path to the Excel file containing image data
    sheet_name : str or int
        Index of the sheet to read
    pixel_num : int
        Column index containing raw pixel count values
    answer_num : int
        Column index containing correctness labels (0/1)

    Returns
    -------
    df : pandas.DataFrame
        Processed DataFrame with columns:
            - answer_4o : Binary correctness labels
            - pixel : Normalized pixel counts (divided by 100,000)
            - pixel_group : Grouped pixel values (0, 2, 4, ..., 30)
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1)
    df['answer_4o'] = df.iloc[:, answer_num].astype(int)
    df['pixel']=df.iloc[:,pixel_num].apply(lambda x: x / 1e5)
    df['pixel_group']=(df['pixel'].astype(int)// 2) * 2
    df.loc[df['pixel_group'] > 30, 'pixel_group'] = 30
    return df

def combine_pixel_df(df1,df2,df3):
    """
    Combine pixel count DataFrames from multiple sources into a single dataset.

    Simple concatenation of relevant columns from three DataFrames for cross-source analysis.

    Parameters
    ----------
    df1, df2, df3 : pandas.DataFrame
        Three DataFrames containing pixel group analysis results.
        Each must contain columns:
            - answer_4o : Binary correctness labels
            - pixel_group : Grouped pixel values

    Returns
    -------
    df : pandas.DataFrame
        Combined DataFrame containing all rows from the three inputs,
        with columns:
            - answer_4o : Binary correctness labels
            - pixel_group : Grouped pixel values
    """
    df = pd.concat(
        [df1[['answer_4o', 'pixel_group']],
         df2[['answer_4o', 'pixel_group']],
         df3[['answer_4o', 'pixel_group']]],
        ignore_index=True)
    return df

def main():
    Lancet_df = get_image_pixel_acc('label summary.xlsx',0,11,14)
    Lancet_result_image = image_pixel_group(Lancet_df)
    NEJM_df = get_image_pixel_acc('label summary.xlsx',1,11,14)
    NEJM_result_image = image_pixel_group(NEJM_df)
    JAMA_df = get_image_pixel_acc('label summary.xlsx',2,11,14)
    JAMA_result_image = image_pixel_group(JAMA_df)
    all_df = combine_pixel_df(NEJM_df, Lancet_df, JAMA_df)
    all_result_image=image_pixel_group(all_df)

if __name__ == '__main__':
    main()
