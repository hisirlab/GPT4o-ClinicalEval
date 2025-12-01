import os
import re
import pandas as pd
import json

def date_data(excel_file,sheet_name,answer_num,date_num):
    """
    Load answer correctness and date information from an Excel sheet,
    compute yearly accuracy and case counts, and compute aggregated
    statistics based on 3-year intervals.

    Parameters
    ----------
    excel_file : str
        Path to the Excel file containing the date and correctness labels for each case.
    sheet_name : str or int
        Excel sheet name or index to read.
    answer_num : int
        Column index of the model correctness indicator (0=wrong, 1=correct).
    date_num : int
        Column index of the date field in each sheet.

    Returns
    -------
    df_merge : pandas.DataFrame
        A DataFrame containing:
            - year_range : 3-year grouped interval (e.g., "2010-2012")
            - accuracy : mean accuracy within the interval
            - case_count : total number of cases in the interval
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1)
    df.iloc[:,answer_num] = pd.to_numeric(df.iloc[:, answer_num], errors="coerce")
    df['accuracy'] = df.iloc[:, answer_num].astype(int)
    df["date"] = pd.to_datetime(df.iloc[:, date_num], errors='coerce')
    df["year"] = df["date"].apply(lambda x: str(x)[-4:] if pd.notnull(x) and str(x)[-4:].isdigit() else None)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Compute mean and standard deviation of valid years
    mean_year = df["year"].mean()
    std_year = df["year"].std()
    print(sheet_name, "mean:", mean_year)
    print(sheet_name, "std:", std_year)

    # Compute case count and accuracy per year
    case_count = df.groupby("year").size().reset_index(name="case_count")
    accuracy = df.groupby("year")["accuracy"].mean().reset_index()
    result = pd.merge(accuracy, case_count, on="year")
    result.columns = ["year", "accuracy", "case_count"]
    print(result)

    # Group years into 3-year intervals: e.g., 2001–2003, 2004–2006
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["year_group"] = (df["year"] // 3) * 3
    df["year_range"] = df["year_group"].astype(str) + "-" + (df["year_group"] + 2).astype(str)
    case_count = df.groupby("year_range").size().reset_index(name="case_count")
    accuracy = df.groupby("year_range")["accuracy"].mean().reset_index()
    df = pd.merge(accuracy, case_count, on="year_range")
    df_merge = df.groupby("year_range").agg({
        "accuracy": "mean",
        "case_count": "sum"
    }).reset_index()
    df_merge["start_year"] = df_merge["year_range"].str.extract(r"(\d+)").astype(int)
    df_merge = df_merge.sort_values("start_year").drop(columns=["start_year"])
    print(df_merge)
    return df_merge

def date_data_all(excel_file,sheet_name1,sheet_name2,sheet_name3,answer_num1,answer_num2,answer_num3,date_num1,date_num2,date_num3):
    """
    Load model answer correctness and date information from three Excel sheets,
    merge them into a single dataset, and compute yearly and 3-year–interval
    accuracy statistics.

    Parameters
    ----------
    excel_file : str
        Path to the Excel file containing the date and correctness labels for each case.
    sheet_name1, sheet_name2, sheet_name3 : str or int
        Names or indices of the three sheets to load.
    answer_num1, answer_num2, answer_num3 : int
        Column indices for correctness labels in each sheet.
    date_num1, date_num2, date_num3 : int
        Column indices containing date information in each sheet.

    Returns
    -------
    df_merge : pandas.DataFrame
        A DataFrame with aggregated accuracy and case count for each
        3-year interval, sorted chronologically.
    """
    sheet_names = [sheet_name1, sheet_name2, sheet_name3]
    dfs = [pd.read_excel(excel_file, sheet_name=sheet_name, header=None, skiprows=1) for sheet_name in sheet_names]
    df1, df2, df3 = dfs
    dfs = []
    for i, (df, answer_num,date_num) in enumerate(zip([df1, df2, df3],[answer_num1, answer_num2, answer_num3],[date_num1,date_num2,date_num3])):
        df["date"] = pd.to_datetime(df.iloc[:, date_num], errors='coerce')
        df["year"] = df["date"].apply(lambda x: str(x)[-4:] if pd.notnull(x) and str(x)[-4:].isdigit() else None)
        df.iloc[:, answer_num] = pd.to_numeric(df.iloc[:, answer_num], errors="coerce")
        df['accuracy'] = df.iloc[:, answer_num].astype(int)
        dfs.append(df)
    df = pd.concat([df[["year", "accuracy"]] for df in dfs], ignore_index=True)

    # Compute mean and standard deviation of valid years
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    mean_year = df["year"].mean()
    std_year = df["year"].std()
    print("mean:", mean_year)
    print("std:", std_year)

    # Compute case count and accuracy per year
    case_count = df.groupby("year").size().reset_index(name="case_count")
    accuracy = df.groupby("year")["accuracy"].mean().reset_index()
    result = pd.merge(accuracy, case_count, on="year")
    result.columns = ["year", "accuracy", "case_count"]
    print(result)

    # Group years into 3-year intervals: e.g., 2001–2003, 2004–2006
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["year_group"] = (df["year"] // 3) * 3
    df["year_range"] = df["year_group"].astype(str) + "-" + (df["year_group"] + 2).astype(str)
    case_count = df.groupby("year_range").size().reset_index(name="case_count")
    accuracy = df.groupby("year_range")["accuracy"].mean().reset_index()
    df = pd.merge(accuracy, case_count, on="year_range")
    df_merge = df.groupby("year_range").agg({
        "accuracy": "mean",
        "case_count": "sum"
    }).reset_index()
    df_merge["start_year"] = df_merge["year_range"].str.extract(r"(\d+)").astype(int)
    df_merge = df_merge.sort_values("start_year").drop(columns=["start_year"])

    print(df_merge)
    return df_merge


def main():
    result_lancet = date_data('label summary.xlsx',0, 14,12)
    result_nejm = date_data('label summary.xlsx',1, 14,12)
    result_jama = date_data('label summary.xlsx',2, 14,12)
    result_all = date_data_all('label summary.xlsx',0,1,2,14,14,14,12,12,12)


if __name__ == "__main__":
    main()