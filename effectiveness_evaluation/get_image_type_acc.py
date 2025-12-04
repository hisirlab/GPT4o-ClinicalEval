import pandas as pd

def image_type_count_acc_data(file_path,sheet_name,image_modality_num,answer_num):
    """
    Compute accuracy statistics based on image type count groups from a given Excel sheet.

    Parameters
    ----------
    file_path : str
        Path to the Excel file that contains the image type and correctness labels.
    sheet_name : int
        Sheet index in the Excel file to be processed.
    image_modality_num : int
        Column index of the image type label for each case in the sheet.
    answer_num : int
        Column index of the correctness indicator (0 = incorrect, 1 = correct).

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing accuracy analysis by sex group, including:
            - image_modality : image type count label
            - accuracy : Mean accuracy for each image type count group
            - case_count : Total number of cases for each image type count group
            - right_case_count : Number of correctly diagnosed cases (accuracy == 1)
            - wrong_case_count : Number of incorrectly diagnosed cases (accuracy == 0)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    df['image_modality'] = df.iloc[:, image_modality_num]
    df["image_modality"] = df["image_modality"].astype(str).fillna("").str.replace(";", ",").apply(lambda x: set(x.split(",")))
    df["image_modality"] = df["image_modality"].apply(lambda x: "Unimodal" if len(x) == 1 else "Multimodal")
    df['accuracy'] = df.iloc[:, answer_num].apply(lambda x: 1 if x == 0 else 0)
    accuracy_by_image_modality = df.groupby("image_modality")["accuracy"].mean().reset_index()
    case_count = df.groupby('image_modality').size().reset_index(name='case_count')
    right_case_count = df.groupby('image_modality')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_image_modality, case_count, on="image_modality")
    result = pd.merge(result, right_case_count, on="image_modality")
    result = result[["image_modality", "accuracy", "case_count","right_case_count"]]
    result.columns = ["image_modality", "accuracy", "case_count","right_case_count"]
    print(result)
    return result

def image_type_count_acc_all_data(file_path,image_modality_num1,image_modality_num2,image_modality_num3,answer_num1,answer_num2,answer_num3):
    """
    Compute accuracy statistics based on image type count groups from three Excel sheets,
    merge them into a single dataset, and compute accuracy statistics.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    for df,image_modality_num,answer_num in zip([df1,df2,df3],[image_modality_num1,image_modality_num2,image_modality_num3],[answer_num1,answer_num2,answer_num3]):
        df['image_modality'] = df.iloc[:, image_modality_num1]
        df["image_modality"] = df["image_modality"].astype(str).fillna("").str.replace(";", ",").apply(lambda x: set(x.split(",")))
        df["image_modality"] = df["image_modality"].apply(lambda x: "Unimodal" if len(x) == 1 else "Multimodal")
        df['accuracy'] = df.iloc[:, answer_num].apply(lambda x: 1 if x == 0 else 0)
    df = pd.concat(
        [df1[['image_modality', 'accuracy']], df2[['image_modality', 'accuracy']], df3[['image_modality', 'accuracy']]],
        ignore_index=True)
    accuracy_by_image_modality = df.groupby("image_modality")["accuracy"].mean().reset_index()
    case_count = df.groupby('image_modality').size().reset_index(name='case_count')
    right_case_count = df.groupby('image_modality')["accuracy"].sum()
    right_case_count = right_case_count.reset_index(name="right_case_count")
    result = pd.merge(accuracy_by_image_modality, case_count, on="image_modality")
    result = pd.merge(result, right_case_count, on="image_modality")
    result = result[["image_modality", "accuracy", "case_count", "right_case_count"]]
    result.columns = ["image_modality", "accuracy", "case_count", "right_case_count"]
    print(result)
    return result

def image_type_weight_acc_data(file_path, sheet_name,image_modality_num,answer_num):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    expanded_rows = []
    for _, row in df.iterrows():
        modalities = str(row[image_modality_num]).split(';')
        case_accuracy = row[answer_num]
        n_images = len(modalities)
        per_image_acc = case_accuracy / n_images if n_images > 0 else 0
        weight = 1 / n_images
        for i in range(n_images):
            modality = modalities[i].strip()
            expanded_rows.append({
                "CaseID": row[0],
                "image_modality": modality,
                "accuracy": per_image_acc,
                "weight": weight
            })
    expanded_df = pd.DataFrame(expanded_rows)
    print(expanded_df)
    group = expanded_df.groupby(["image_modality"])
    case_count = group["weight"].sum().reset_index(name="case_count")
    right_count = group["accuracy"].sum().reset_index(name="right_case_count")
    result = case_count.merge(right_count, on=["image_modality"])
    result["accuracy"] = result["right_case_count"] / result["case_count"]
    modality_mapping = {
        '0': "Others", '1': "CT", '2': "MRI", '3': "PET", '4': "SPECT",
        '5': "X-ray", '6': "Histol.", '7': "US", '8': "DSC", '9': "EEG",
        '10': "ECG", '11': "OCT", '12': "Endo.", '13': "FP", '14': "DSA",
        '15': "Photo", '16': "Slitlamp", '17': "FA"
    }
    result["image_modality"] = result["image_modality"].replace(modality_mapping)
    result = result[["image_modality", "case_count", "right_case_count"]]
    result = result.sort_values(by=["image_modality", "case_count"],ascending=[True, False]).reset_index(drop=True)
    print(result)
    return result

def image_type_weight_acc_all_data(file_path,image_modality_num1,image_modality_num2,image_modality_num3,answer_num1,answer_num2,answer_num3):
    """
    Load multimodal medical image data from three Excel sheets,
    calculate image typy-specific accuracy using weighted allocation,
    and aggregate performance statistics across all journals.

    • The total accuracy of each case is evenly distributed among all image entries in that case.
    • Each image entry receives a weight equal to the reciprocal of the total number of image entries in the case (1/n_images).
    • Statistics are computed by aggregating weighted accuracy and weights across all cases,
      then mapping modality codes to descriptive names.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the three journal datasets.
    image_modality_num1, image_modality_num2, image_modality_num3 : int
        Column indices for the image type strings in each sheet.
        Modalities are semicolon-separated strings (e.g., "1;3;5").
    answer_num1, answer_num2, answer_num3 : int
        Column indices for case correctness values (0=wrong, 1=correct)
        in each respective sheet.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing:
            - image_modality : Descriptive name of the medical imaging type
            - accuracy : Image type-specific accuracy (weighted average)
            - case_count : Total weighted case count contributed by the image type
            - right_case_count : Total weighted correct case count for the image type
        Sorted alphabetically by image type name.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    expanded_rows = []
    for df,image_modality_num,answer_num in zip([df1, df2,df3],[image_modality_num1,image_modality_num2,image_modality_num3],[answer_num1, answer_num2, answer_num3]):
        for _, row in df.iterrows():
            modalities = str(row[image_modality_num]).split(';')
            case_accuracy = row[answer_num]
            n_images = len(modalities)
            per_image_acc = case_accuracy / n_images if n_images > 0 else 0
            weight = 1 / n_images
            for i in range(n_images):
                modality = modalities[i].strip()
                expanded_rows.append({
                    "CaseID": row[0],
                    "image_modality": modality,
                    "accuracy": per_image_acc,
                    "weight": weight
                })
    expanded_df = pd.DataFrame(expanded_rows)
    print(expanded_df)

    group = expanded_df.groupby(["image_modality"])
    case_count = group["weight"].sum().reset_index(name="case_count")
    right_count = group["accuracy"].sum().reset_index(name="right_case_count")
    result = case_count.merge(right_count, on=["image_modality"])
    result["accuracy"] = result["right_case_count"] / result["case_count"]
    modality_mapping = {
        '0': "Others", '1': "Computed Tomography", '2': "Magnetic Resonance Imaging", '3': "Positron Emission Tomography (CT/MRI)",
        '4': "Single-Photon Emission Computed Tomography (CT/MRI)",'5': "X-ray", '6': "Histopathological image", '7': "Ultrasound",
        '8': "Dermatoscopy", '9': "Electroencephalography",'10': "Electrocardiography", '11': "Optical Coherence Tomography",
        '12': "Endoscopy", '13': "Fundus photograph", '14': "Digital Subtraction Angiography",
        '15': "Clinical photograph", '16': "Slit-lamp photography", '17': "Fluorescein fundus angiogram"
    }
    result["image_modality"] = result["image_modality"].replace(modality_mapping)
    result = result[["image_modality", "accuracy", "case_count", "right_case_count"]]
    result = result.sort_values(by=["image_modality", "accuracy", "case_count"],ascending=[True, False, False]).reset_index(drop=True)
    print(result)
    return result


def main():
    # -------------- image type count --------------
    Lancet_result=image_type_count_acc_data('label summary.xlsx',0,4,12)
    NEJM_result = image_type_count_acc_data('label summary.xlsx', 1, 4, 12)
    JAMA_result=image_type_count_acc_data('label summary.xlsx',2,4,12)
    All_result=image_type_count_acc_all_data('label summary.xlsx',4,4,4,12,12,12)

    # ----------------- image type -----------------
    Lancet_result = image_type_weight_acc_data('label summary.xlsx', 0, 4, 12)
    NEJM_result = image_type_weight_acc_data('label summary.xlsx', 1, 4, 12)
    JAMA_result = image_type_weight_acc_data('label summary.xlsx', 2, 4, 12)
    All_result = image_type_weight_acc_all_data('label summary.xlsx', 4, 4, 4, 12, 12, 12)


if __name__ == "__main__":
    main()
