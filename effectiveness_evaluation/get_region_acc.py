import pandas as pd

def image_type_region_weighted_acc_data(file_path, sheet_name, image_modality_num, region_num, answer_num):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=1)
    expanded_rows = []
    for _, row in df.iterrows():
        modalities = str(row[image_modality_num]).split(';')
        region_groups = str(row[region_num]).split(';')
        case_accuracy = row[answer_num]
        n_images = len(modalities)
        per_image_acc = case_accuracy / n_images if n_images > 0 else 0
        image_weight=1/n_images
        for i in range(n_images):
            modality = modalities[i].strip()
            region_str = region_groups[i].strip() if i < len(region_groups) else ""
            regions_raw = [r.strip() for r in region_str.split(',') if r.strip()]
            n_regions_in_image = len(regions_raw)
            per_region_acc = per_image_acc / n_regions_in_image
            region_weight = image_weight / n_regions_in_image
            for r in regions_raw:
                region_id = r.split('.')[0]
                expanded_rows.append({
                    "CaseID": row[0],
                    "image_modality": modality,
                    "region": region_id,
                    "accuracy": per_region_acc,
                    "weight": region_weight
                })
    expanded_df = pd.DataFrame(expanded_rows)
    print(expanded_df)
    expanded_df = expanded_df[expanded_df["region"].apply(lambda x: len(x.strip().split(" ")) == 1)]
    expanded_df["region"] = expanded_df["region"].apply(
        lambda x: x.strip() if x.strip() in ["0","1", "2", "3", "4", "5", "6", "7"] else "NA")
    group = expanded_df.groupby(["image_modality", "region"])
    case_count = group["weight"].sum().reset_index(name="case_count")
    right_count = group["accuracy"].sum().reset_index(name="right_case_count")
    result = case_count.merge(right_count, on=["image_modality", "region"])
    result["accuracy"]=result["right_case_count"]/result["case_count"]
    modality_mapping = {
        '0': "Others", '1': "CT", '2': "MRI", '3': "PET", '4': "SPECT",
        '5': "X-ray", '6': "Histol.", '7': "Ultrasound", '8': "DSC", '9': "EEG",
        '10': "ECG", '11': "OCT", '12': "Endoscopy", '13': "FP", '14': "DSA",
        '15': "Clinical photograph", '16': "Slitlamp", '17': "Fluorescein angiogram"
    }
    result["image_modality"] = result["image_modality"].replace(modality_mapping)
    region_mapping = {
        '0': "Others", '1': "Head/Neck", '2': "Thorax", '3': "Abdomen", '4': "Pelv/Perin.", '5': "Upper Limb",
        '6': "Lower Limb", '7': "Back","NA":"N/A"
    }
    result["region"] = result["region"].replace(region_mapping)
    small_cases = result[result['case_count'] < 10]
    large_cases = result[result['case_count'] >= 10]
    if not small_cases.empty:
        other_row = pd.DataFrame({
            "region": ["Others"],
            "accuracy": [small_cases["right_case_count"].sum() / small_cases["case_count"].sum()],
            "case_count": [small_cases["case_count"].sum()],
            "right_case_count": [small_cases["right_case_count"].sum()]
        })
        result = pd.concat([large_cases, other_row], ignore_index=True)
    result = result[result["image_modality"] != "Others"]
    result = result[~result["image_modality"].isin(["PET", "SPECT", "Histol.", "DSC", "EEG", "ECG", "OCT", "FP", "DSA", "Slitlamp", "Fluorescein angiogram"])]
    result = result[["image_modality", "region", "accuracy", "case_count", "right_case_count"]]
    result = result.sort_values(by=["image_modality", "region", "accuracy", "case_count"], ascending=[True, True, False, False]).reset_index(drop=True)
    print(result)
    return result

def image_type_region_weighted_acc_all_data(file_path, image_modality_num1,image_modality_num2,image_modality_num3, region_num1,region_num2,region_num3, answer_num1, answer_num2, answer_num3):
    """
    Calculate accuracy metrics for medical imaging types across different anatomical regions,
    using a hierarchical weighting system that accounts for both image-level and region-level contributions.

    1. Data Loading: Read three Excel sheets containing case-level data
    2. Hierarchical Expansion: Split multi-modality, multi-region cases into individual region entries
    3. Weight Allocation:
       - Image-level weight: 1/(number of images in case)
       - Region-level weight: image_weight/(number of regions in image)
    4. Accuracy Allocation: Case accuracy distributed proportionally through the hierarchy
    5. Region Standardization: Clean and categorize anatomical regions
    6. Aggregation: Group by modality-region pairs to calculate weighted statistics
    7. Small Group Handling: Merge low-count regions (<10 cases) into "Others" category
    8. Modality Filtering: Remove uncommon modalities and "Others" modality category

    Weighting Methodology
    ---------------------
    • Each case's total weight = 1.0 (distributed across all regions in all images)
    • For a case with N images, each image gets weight = 1/N
    • For an image with M regions, each region gets weight = (1/N)/M
    • Accuracy follows the same proportional allocation

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the three journal datasets.
    image_modality_num1, image_modality_num2, image_modality_num3 : int
        Column indices for image modality strings (semicolon-separated) in each sheet.
    region_num1, region_num2, region_num3 : int
        Column indices for anatomical region strings (semicolon-separated, comma-separated within) in each sheet.
    answer_num1, answer_num2, answer_num3 : int
        Column indices for correctness values (0=correct, 1=incorrect) in each sheet.

    Returns
    -------
    result : pandas.DataFrame
        A DataFrame containing:
            - image_modality : Shortened modality name (e.g., "CT", "MRI")
            - region : Anatomical region name (e.g., "Head/neck", "Thorax")
            - accuracy : Weighted accuracy for the modality-region combination
            - case_count : Weighted case count
            - right_case_count : Weighted count of correctly predicted cases
        Sorted by image type, region, then accuracy.
    """
    df1 = pd.read_excel(file_path, sheet_name=0, header=None, skiprows=1)
    df2 = pd.read_excel(file_path, sheet_name=1, header=None, skiprows=1)
    df3 = pd.read_excel(file_path, sheet_name=2, header=None, skiprows=1)
    expanded_rows = []
    for df,image_modality_num,region_num,answer_num in zip([df1, df2,df3],[image_modality_num1,image_modality_num2,image_modality_num3],[region_num1,region_num2,region_num3],[answer_num1, answer_num2, answer_num3]):
        for _, row in df.iterrows():
            modalities = str(row[image_modality_num]).split(';')
            region_groups = str(row[region_num]).split(';')
            case_accuracy = row[answer_num]
            n_images = len(modalities)
            per_image_acc = case_accuracy / n_images if n_images > 0 else 0
            image_weight = 1.0 / n_images
            for i in range(n_images):
                modality = modalities[i].strip()
                region_str = region_groups[i].strip() if i < len(region_groups) else ""
                regions_raw = [r.strip() for r in region_str.split(',') if r.strip()]
                n_regions_in_image = len(regions_raw)
                per_region_acc = per_image_acc / n_regions_in_image
                region_weight = image_weight / n_regions_in_image
                for r in regions_raw:
                    region_id = r.split('.')[0]
                    expanded_rows.append({
                        "CaseID": row[0],
                        "image_modality": modality,
                        "region": region_id,
                        "accuracy": per_region_acc,
                        "weight": region_weight
                    })
    expanded_df = pd.DataFrame(expanded_rows)
    print(expanded_df)
    expanded_df = expanded_df[expanded_df["region"].apply(lambda x: len(x.strip().split(" ")) == 1)]
    expanded_df["region"] = expanded_df["region"].apply(
        lambda x: x.strip() if x.strip() in ["0","1", "2", "3", "4", "5", "6", "7"] else "NA")
    group = expanded_df.groupby(["image_modality", "region"])
    case_count = group["weight"].sum().reset_index(name="case_count")
    right_count = group["accuracy"].sum().reset_index(name="right_case_count")
    result = case_count.merge(right_count, on=["image_modality", "region"])
    result["accuracy"] = result["right_case_count"] / result["case_count"]
    modality_mapping = {
        '0': "Others", '1': "CT", '2': "MRI", '3': "PET", '4': "SPECT",
        '5': "X-ray", '6': "Histol.", '7': "Ultrasound", '8': "DSC", '9': "EEG",
        '10': "ECG", '11': "OCT", '12': "Endoscopy", '13': "FP", '14': "DSA",
        '15': "Clinical photograph", '16': "Slitlamp", '17': "Fluorescein angiogram"
    }
    result["image_modality"] = result["image_modality"].replace(modality_mapping)
    region_mapping = {
        '0': "Others", '1': "Head/neck", '2': "Thorax", '3': "Abdomen", '4': "Pelv/perin.", '5': "Upper limb",
        '6': "Lower limb", '7': "Back","NA":"N/A"
    }
    result["region"] = result["region"].replace(region_mapping)

    final_result=pd.DataFrame()
    image_modality_groups=result.groupby("image_modality")
    for group_name, group_data in image_modality_groups:
        small_cases = group_data[
            (group_data['case_count'] < 10) |
            (group_data["region"] == "Others")
            ]
        large_cases = group_data[
            (group_data['case_count'] >= 10) &
            (group_data["region"] != "Others")
            ]
        if not small_cases.empty:
            other_row = pd.DataFrame({
                "image_modality": [group_name],
                "region": ["Others"],
                "accuracy": [small_cases["right_case_count"].sum() / small_cases["case_count"].sum()],
                "case_count": [small_cases["case_count"].sum()],
                "right_case_count": [small_cases["right_case_count"].sum()]
            })
            group_result = pd.concat([large_cases, other_row], ignore_index=True)
        else:
            group_result = large_cases
        final_result = pd.concat([final_result, group_result], ignore_index=True)
    result=final_result
    result = result[result["image_modality"] != "Others"]
    result=result[~result["image_modality"].isin(["PET", "SPECT","Histol.","DSC", "EEG", "ECG","OCT","FP","DSA","Slitlamp","Fluorescein angiogram"])]
    result = result[["image_modality", "region", "accuracy", "case_count", "right_case_count"]]
    result = result.sort_values(by=["image_modality", "region", "accuracy", "case_count"],
                                ascending=[True, True, False, False]).reset_index(drop=True)
    print(result)
    return result

def main ():
    Lancet_result = image_type_region_weighted_acc_data('label summary.xlsx',0,4,10,12)
    NEJM_result = image_type_region_weighted_acc_data('label summary.xlsx', 0,4,10,12)
    JAMA_result = image_type_region_weighted_acc_data('label summary.xlsx', 0,4,10,12)
    All_result = image_type_region_weighted_acc_all_data('label summary.xlsx', 4,4,4, 10, 10, 10,12, 12, 12)

if __name__ == "__main__":
    main()