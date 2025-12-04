import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

import get_dataset_acc
import get_date_acc
import get_sex_acc
import get_age_acc
import get_task_type_acc
import get_image_type_acc
import get_image_count_acc
import get_icd_acc
import get_specialty_acc
import get_region_acc
import get_word_count_acc

def get_overall_acc_figure():
    """
    Generate a grouped bar chart comparing accuracy across three journals (Lancet, NEJM, JAMA)
    and their combined summary ("All") before and after the knowledge cutoff date.

    This function:
    1. Calls `dataset_acc_data()` to compute accuracy statistics for:
            - Full dataset
            - Post-cutoff dataset
    2. For each journal (Lancet, NEJM, JAMA, All), two bars are plotted:
            - Full dataset accuracy
            - Post-cutoff accuracy
    3. Draws:
            - Accuracy bars with different colors
            - Numeric accuracy values above bars
            - Correct/total case counts below bars
            - Horizontal markers between paired bars
            - A visual bracket between two bars with placeholder p-values
    4. Formats the figure with journal names, legends, axis labels, and styling.


    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated accuracy comparison figure.

    """

    # Compute accuracy statistics
    all_result = get_dataset_acc.dataset_acc_data('label summary.xlsx',12,14)
    datasets = ['Lancet', 'NEJM', 'JAMA', 'All']
    p_values = ['$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx']
    x = np.arange(len(datasets))
    colors = {'Full': '#9370DB', 'Post': '#FFC2C3'}
    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(15, 10))

    # Draw grouped bars
    for i, dataset in enumerate(datasets):
        sub_df = all_result[all_result['dataset'] == dataset]
        x_positions = []
        heights = []
        for j, post_type in enumerate(['Full', 'Post']):
            row = sub_df[sub_df['post'] == post_type].iloc[0]
            x_pos = i + (j - 0.5) * bar_width
            x_positions.append(x_pos)
            heights.append(row['accuracy'])
            plt.bar(
                x=x_pos,
                height=row['accuracy'],
                width=bar_width * 0.9,
                color=colors[post_type],
                label=post_type if i == 0 else "",
                zorder=1
            )
            plt.text(
                x_pos,
                row['accuracy'] + 0.005,
                f"{row['accuracy']:.2f}",
                ha='center', va='bottom', fontsize=24, color=colors[post_type]
            )
            count = f"{int(row['right_case_count'])}\n{int(row['case_count'])}"
            plt.text(
                x_pos,
                row['accuracy'] - 0.09,
                count,
                ha='center', va='bottom', fontsize=22, color="black"
            )
            ax.hlines(y=row['accuracy'] - 0.045,
                      xmin=x_pos - bar_width/2 +0.03, xmax=x_pos + bar_width/2 - 0.03, color='black',
                      linewidth=2
            )
        y_top = max(heights) + 0.08
        bracket_height = 0.02
        ax.hlines(y=y_top, xmin=x_positions[0], xmax=x_positions[1], color='black', linewidth=2)
        ax.vlines(x=x_positions[0], ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
        ax.vlines(x=x_positions[1], ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
        ax.text(np.mean(x_positions), y_top + 0.005, p_values[i],ha='center', va='bottom', fontsize=22)

    # Legend and axis formatting
    plt.xticks(range(len(datasets)), datasets)
    full_marker = plt.Line2D([], [], color=colors['Full'], marker='s', linestyle='None', markersize=20,
                             label='Full dataset')
    post_marker = plt.Line2D([], [], color=colors['Post'], marker='s', linestyle='None', markersize=20,
                             label='Post-cutoff dataset')
    plt.legend(
        handles=[full_marker, post_marker],
        fontsize=28,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        frameon=False
    )
    plt.ylabel('Accuracy', fontsize=30, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=28)
    ax.set_xlabel("Journal", fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=28)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.show()
    return fig

def get_date_figure():
    """
    Generate a combined scatter-and-line plot illustrating the temporal trend
    of model accuracy across different journals (Lancet, NEJM, JAMA)
    and the merged dataset (“All”).

    Visualization:
    -------------
    • X-axis : 3-year time intervals (e.g., 2010–2012)
    • Y-axis : Mean accuracy
    • Line   : Accuracy trend for each dataset
    • Dot    : Accuracy at each interval
    • Dot size : Proportional to the number of cases in that interval

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated plot figure.
    """
    result_lancet = get_date_acc.date_data('label summary.xlsx',0, 14,12)
    result_lancet = result_lancet.dropna(subset=['year_range'])
    result_nejm = get_date_acc.date_data('label summary.xlsx',1, 14,12)
    result_nejm = result_nejm.dropna(subset=['year_range'])
    result_jama = get_date_acc.date_data('label summary.xlsx',2, 14,12)
    result_jama = result_jama.dropna(subset=['year_range'])
    result_all = get_date_acc.date_data_all('label summary.xlsx',0,1,2,14,14,14,12,12,12)
    result_all = result_all.dropna(subset=['year_range'])

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(result_lancet['year_range'].astype(str), result_lancet['accuracy'], marker='.', color='#6A994E',
             linestyle='--', linewidth=2, label='Lancet  ($p$=0.xx)', zorder=1)
    plt.scatter(result_lancet['year_range'].astype(str), result_lancet['accuracy'], s=result_lancet['case_count'] * 8,
                color='#6A994E', alpha=0.5, zorder=1)

    plt.plot(result_nejm['year_range'].astype(str), result_nejm['accuracy'], marker='.', color='#FFA94D',
             linestyle='--', linewidth=2, label='NEJM  ($p$=0.xx)', zorder=2)
    plt.scatter(result_nejm['year_range'].astype(str), result_nejm['accuracy'], s=result_nejm['case_count'] * 8,
                color='#FFA94D', alpha=0.5, zorder=2)

    plt.plot(result_jama['year_range'].astype(str), result_jama['accuracy'], marker='.', color='#355C7D', linestyle='--',
            linewidth=2, label='JAMA  ($p$=0.xx)', zorder=3)
    plt.scatter(result_jama['year_range'].astype(str), result_jama['accuracy'], s=result_jama['case_count'] * 8,
                color='#355C7D', alpha=0.5, zorder=3)

    plt.plot(result_all['year_range'].astype(str), result_all['accuracy'], marker='.', color='#BC4749',
             label='All  ($p$=0.xx)', zorder=4)
    plt.scatter(result_all['year_range'].astype(str), result_all['accuracy'], s=result_all['case_count'] * 8,
                color='#BC4749', alpha=0.7, zorder=4)

    plt.ylim(0, 1)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=26,rotation=18)
    plt.xlabel("Year", fontsize=30, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=30, fontweight='bold')
    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ['Lancet  ($p$=0.xx)','NEJM  ($p$=0.xx)', 'JAMA  ($p$=0.xx)', 'All  ($p$=0.xx)']
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    plt.legend(sorted_handles, sorted_labels, loc='lower left', fontsize=27, frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_sex_figure():
    """
    Generate a grouped bar chart comparing model accuracy by sex
    across different journals (Lancet, NEJM, JAMA) and the merged dataset (“All”).

    Data Source
    -----------
    Extracted by calling:
        - get_sex_acc.sex_acc_data()
        - get_sex_acc.sex_acc_data_all()

    Visualization
    ---------------------
    • axis : Journal datasets (Lancet, NEJM, JAMA, All)
    • Grouped bars : Sex categories (Male, Female, Unclear)
    • Y-axis : Accuracy (0–1 scale)
    • Annotation :
        - Accuracy value above each bar
        - "Correct cases / Total cases" below each bar
        - Brackets showing statistical comparison among sex groups

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further saving or customization.
    """
    unclear = []
    female = []
    male = []
    Lancet_result = get_sex_acc.sex_acc_data('label summary.xlsx',0, 6,12)
    NEJM_result = get_sex_acc.sex_acc_data('label summary.xlsx',0, 6,12)
    JAMA_result = get_sex_acc.sex_acc_data('label summary.xlsx',0, 6,12)
    All_result = get_sex_acc.sex_acc_data_all('label summary.xlsx',6,6,6,12,12,12)
    for i,sex_group in enumerate([unclear,female,male]):
        for journal in [Lancet_result,NEJM_result,JAMA_result,All_result]:
            sex_group.append(journal.iloc[i])
    datasets = ['Lancet', 'NEJM', 'JAMA', 'All']
    p_values_sex = ['$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx']
    p_values_missing = ['$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx']
    bar_width = 0.3
    group_width = 1.2
    colors = {'Male': '#4C72B0', 'Female': '#DD8452', 'Unclear':'#9CAF88'}

    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(len(datasets)):
        group_center = i * group_width
        for sex_group,color_group,shift in zip([male,female,unclear],['Male','Female','Unclear'],[-1,0,1]):
            x=group_center+shift*bar_width
            ax.bar(x, sex_group[i]['accuracy'], width=bar_width, color=colors[color_group])
            ax.text(x, sex_group[i]['accuracy'] + 0.01, f"{sex_group[i]['accuracy']:.2f}", ha='center', fontsize=24)
            ax.text(x, sex_group[i]['accuracy'] - 0.09, f"{sex_group[i]['right_case_count']}\n{sex_group[i]['case_count']}", ha='center',
                    fontsize=22)
            ax.hlines(sex_group[i]['accuracy'] - 0.05,
                      xmin=x - bar_width / 2 + 0.03, xmax=x + bar_width / 2 - 0.03, color='black',
                      linewidth=2
                      )
        x_male = group_center - bar_width
        y_male = male[i]['accuracy']
        x_female = group_center
        y_female = female[i]['accuracy']
        x_unclear = group_center + bar_width
        y_unclear = unclear[i]['accuracy']
        y_top_sex = max(y_male, y_female) + 0.08
        bracket_height = 0.02
        ax.hlines(y=y_top_sex, xmin=x_male, xmax=x_female, color='black', linewidth=2)
        ax.vlines(x=x_male, ymin=y_top_sex - bracket_height, ymax=y_top_sex, color='black', linewidth=2)
        ax.vlines(x=x_female, ymin=y_top_sex - bracket_height, ymax=y_top_sex, color='black', linewidth=2)
        ax.text((x_male + x_female) / 2, y_top_sex + 0.005, p_values_sex[i], ha='center', va='bottom', fontsize=22)

        y_top_missing = max(y_male, y_female, y_unclear) + 0.14
        ax.hlines(y=y_top_missing, xmin=x_male, xmax=x_female, color='black', linewidth=2)
        ax.hlines(y=y_top_missing+bracket_height, xmin=x_male+bar_width/2, xmax=x_unclear, color='black', linewidth=2)
        ax.vlines(x=x_male+bar_width/2, ymin=y_top_missing, ymax=y_top_missing+bracket_height, color='black', linewidth=2)
        ax.vlines(x=x_unclear, ymin=y_top_missing, ymax=y_top_missing+bracket_height, color='black', linewidth=2)
        ax.text((x_male + x_female + x_unclear) / 3, y_top_missing+bracket_height + 0.005, p_values_missing[i],
                ha='center', va='bottom', fontsize=22)

    xtick_positions = [i * group_width for i in range(len(datasets))]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(datasets, fontsize=26)
    legend_handles = [
        plt.Line2D([0], [0], color=colors['Male'], label='Male', marker='s', linestyle='None', markersize=28),
        plt.Line2D([0], [0], color=colors['Female'], label='Female', marker='s', linestyle='None',markersize=28),
        plt.Line2D([0], [0], color=colors['Unclear'], label='N/A', marker='s', linestyle='None', markersize=28)
    ]
    ax.legend(handles=legend_handles, fontsize=28, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    ax.set_ylabel('Accuracy', fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlabel("Journal", fontsize=30, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig

def get_age_figure():
    """
    Generate a bar chart visualizing model accuracy across different age groups,
    with case counts encoded through a continuous color scale.

    Data Source
    -----------
    Extracted by calling:
        - get_age_acc.age_groups_acc_all_data()

    Visualization
    ---------------------
    • X-axis : Age groups
    • Y-axis : Accuracy (0–1 scale)
    • Bar color : Number of cases per age group (Purple gradient colormap)
    • Annotation :
        - Accuracy value above each bar
        - "Correct cases / Total cases" displayed inside each bar
        - Horizontal divider separating accuracy from counts text
        - Statistical brackets indicating group comparisons with p-values

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for saving or further customization.
    """
    all_result = get_age_acc.age_groups_acc_all_data('label summary.xlsx',5, 5,5,12,12,12)
    age_group = np.array(all_result['age_group'])
    accuracy = np.array(all_result['accuracy'])
    case_count = np.array(all_result['case_count'])
    right_case_count = np.array(all_result['right_case_count'])
    p_value_age= '$p$ = 0.xx'
    p_value_missing = '$p$ = 0.xx'
    x=np.arange(len(age_group))
    group_start_vals = np.array([g for g in case_count])
    norm = mcolors.Normalize(vmin=group_start_vals.min(), vmax=group_start_vals.max())
    cmap = ListedColormap(cm.get_cmap('Purples')(np.linspace(0.2, 0.7, 256)))
    bar_colors = []
    for i, val in enumerate(case_count):
        bar_colors.append(cmap(norm(val)))
    fig, ax = plt.subplots(figsize=(15,10))
    bar_width=0.6
    bars = ax.bar(x, accuracy, bar_width, color=bar_colors)
    for i,bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x()+ + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=24, color='black')
        ax.text(bar.get_x() + + bar.get_width() / 2, height - 0.09, f'{right_case_count[i]}\n{case_count[i]}',
                ha='center', va='bottom', fontsize=22, color='black')
        ax.hlines(y=height - 0.045, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                  linewidth=2)
    plt.ylim(0, 1.05)
    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.5])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Case Count', fontsize=25)
    cbar.ax.tick_params(labelsize=17)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(age_group, fontsize=26)
    ax.set_xlabel("Age", fontsize=30, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=30, fontweight='bold')

    # === Statistical comparison annotations === #
    bracket_height = 0.02
    max_y = accuracy.max()
    # First bracket: compares all special groups
    x1, y1 = x[0], accuracy[0]
    x2, y2 = x[-2], accuracy[-2]
    x3, y3 = x[-1], accuracy[-1]
    y_top1 = max_y + 0.08
    ax.hlines(y=y_top1, xmin=x1, xmax=x2, color='black', linewidth=2)
    ax.vlines(x=x1, ymin=y_top1 - bracket_height, ymax=y_top1, color='black', linewidth=2)
    ax.vlines(x=x2, ymin=y_top1 - bracket_height, ymax=y_top1, color='black', linewidth=2)
    ax.text((x1 + x2) / 2, y_top1 + 0.005, p_value_age, ha='center', va='bottom', fontsize=22)

    # Second bracket: compares missing or special last group
    y_top2 = max_y + 0.14
    ax.hlines(y=y_top2, xmin=x1, xmax=x2, color='black', linewidth=2)
    ax.hlines(y=y_top2 + bracket_height, xmin=(x1+x2) / 2, xmax=x3, color='black', linewidth=2)
    ax.vlines(x=(x1+x2) / 2, ymin=y_top2, ymax=y_top2 + bracket_height, color='black',linewidth=2)
    ax.vlines(x=x3, ymin=y_top2, ymax=y_top2 + bracket_height, color='black', linewidth=2)
    ax.text(((x1+x2) / 2 + x3) / 2, y_top2 + bracket_height + 0.005, p_value_missing, ha='center', va='bottom', fontsize=22)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_task_type_figure():
    """
    Generate a grouped bar chart comparing model accuracy by task type
    across different journals (Lancet, NEJM, JAMA) and the merged dataset (“All”).

    Data Source
    -----------
    Extracted by calling:
        - get_task_type_acc.task_type_acc_data
        - get_task_type_acc.task_type_acc_all_data

    Visualization
    ---------------------
    • axis : Journal datasets (Lancet, NEJM, JAMA, All)
    • Grouped bars : Task type categories (Diagnosis, Non-diagnosis)
    • Y-axis : Accuracy (0–1 scale)
    • Annotation :
        - Accuracy value above each bar
        - "Correct cases / Total cases" below each bar
        - Brackets showing statistical comparison among task type groups

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further saving or customization.
    """
    diagnosis=[]
    non_diagnosis = []
    Lancet_result = get_task_type_acc.task_type_acc_data('label summary.xlsx',0,7,12)
    NEJM_result = get_task_type_acc.task_type_acc_data('label summary.xlsx',1,7,12)
    JAMA_result = get_task_type_acc.task_type_acc_data('label summary.xlsx',2,7,12)
    All_result = get_task_type_acc.task_type_acc_all_data('label summary.xlsx',7,7,7,12,12,12)
    for i,task in enumerate([diagnosis,non_diagnosis]):
        for journal in [Lancet_result,NEJM_result,JAMA_result,All_result]:
            task.append(journal[i])
    datasets = ['Lancet', 'NEJM', 'JAMA', 'All']
    p_values = ['$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx']
    bar_width = 0.4
    group_width = 1.2
    colors = {'Diagnosis': '#4C72B0', 'Non-diagnosis': '#DD8452'}
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(len(datasets)):
        group_center = i * group_width
        for task,shift,group_color in zip([diagnosis,non_diagnosis],[-1,1],['Diagnosis', 'Non-diagnosis']):
            x = group_center + shift * (bar_width / 2)
            ax.bar(x, task[i]['accuracy'], width=bar_width, color=colors[group_color])
            ax.text(x, task[i]['accuracy'] + 0.01, f"{task[i]['accuracy']:.2f}", ha='center', fontsize=24)
            ax.text(x, task[i]['accuracy'] - 0.08, f"{task[i]['right_case_count']}\n{task[i]['case_count']}",
                    ha='center',
                    fontsize=22)
            ax.hlines(task[i]['accuracy'] - 0.05,
                      xmin=x - bar_width / 2 + 0.03, xmax=x + bar_width / 2 - 0.03, color='black',
                      linewidth=2
                      )
        x_diagnosis = group_center - bar_width / 2
        x_non_diag = group_center + bar_width / 2
        y_diag = diagnosis[i]['accuracy']
        y_non_diag = non_diagnosis[i]['accuracy']
        y_top_diag = max(y_diag, y_non_diag) + 0.08
        bracket_height = 0.02
        ax.hlines(y=y_top_diag, xmin=x_diagnosis, xmax=x_non_diag, color='black', linewidth=2)
        ax.vlines(x=x_diagnosis, ymin=y_top_diag - bracket_height, ymax=y_top_diag, color='black', linewidth=2)
        ax.vlines(x=x_non_diag, ymin=y_top_diag - bracket_height, ymax=y_top_diag, color='black', linewidth=2)
        ax.text((x_diagnosis + x_non_diag) / 2, y_top_diag + 0.005, p_values[i], ha='center', va='bottom', fontsize=22)
    xtick_positions = [i * group_width for i in range(len(datasets))]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(datasets, fontsize=26)
    legend_handles = [
        plt.Line2D([0], [0], color=colors['Diagnosis'], label='Diagnosis',marker='s', linestyle='None', markersize=20),
        plt.Line2D([0], [0], color=colors['Non-diagnosis'], label='Non-diagnosis',marker='s', linestyle='None', markersize=20),
    ]
    ax.legend(handles=legend_handles, fontsize=28, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    ax.set_ylabel('Accuracy', fontsize=30, fontweight='bold')
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xlabel("Journal", fontsize=30, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()
    return fig

def get_image_type_count_figure():
    """
    Generate a grouped bar chart comparing model accuracy by image type count groups
    across different journals (Lancet, NEJM, JAMA) and the merged dataset (“All”).

    Data Source
    -----------
    Extracted by calling:
        - get_image_type_acc.image_type_count_acc_data
        - get_image_type_acc.image_type_count_acc_all_data

    Visualization
    ---------------------
    • axis : Journal datasets (Lancet, NEJM, JAMA, All)
    • Grouped bars : Image type count categories (Multiple, Single)
    • Y-axis : Accuracy (0–1 scale)
    • Annotation :
        - Accuracy value above each bar
        - "Correct cases / Total cases" below each bar
        - Brackets showing statistical comparison among image type count groups

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object for further saving or customization.
    """
    Lancet_result = get_image_type_acc.image_type_count_acc_data('label summary.xlsx',0,4,12)
    NEJM_result = get_image_type_acc.image_type_count_acc_data('label summary.xlsx',1,4,12)
    JAMA_result = get_image_type_acc.image_type_count_acc_data('label summary.xlsx',2,4,12)
    All_result = get_image_type_acc.image_type_count_acc_all_data('label summary.xlsx',4,4,4,12,12,12)
    datasets = ["Lancet", "NEJM", "JAMA", "All"]
    Multimodal_acc=[Lancet_result['accuracy'][0],NEJM_result['accuracy'][0],JAMA_result['accuracy'][0],All_result['accuracy'][0]]
    Unimodal_acc=[Lancet_result['accuracy'][1],NEJM_result['accuracy'][1],JAMA_result['accuracy'][1],All_result['accuracy'][1]]
    Multimodal_case_count = [Lancet_result['case_count'][0], NEJM_result['case_count'][0], JAMA_result['case_count'][0],All_result['case_count'][0]]
    Unimodal_case_count = [Lancet_result['case_count'][1], NEJM_result['case_count'][1], JAMA_result['case_count'][1],All_result['case_count'][1]]
    Multimodal_right_case_count = [Lancet_result['right_case_count'][0], NEJM_result['right_case_count'][0], JAMA_result['right_case_count'][0],All_result['right_case_count'][0]]
    Unimodal_right_case_count = [Lancet_result['right_case_count'][1],NEJM_result['right_case_count'][1], JAMA_result['right_case_count'][1], All_result['right_case_count'][1]]
    x = np.arange(len(datasets))
    p_values = ['$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx', '$p$ = 0.xx']
    fig, ax = plt.subplots(figsize=(15,10))
    bar_width = 0.35
    colors = {'Multiple': '#4C72B0', 'Single': '#DD8452'}
    bars1 = ax.bar(x - bar_width/2, Multimodal_acc, bar_width, label="Multiple", color=colors['Multiple'])
    bars2 = ax.bar(x + bar_width/2, Unimodal_acc, bar_width, label="Single", color=colors['Single'])
    for bars,case_count,right_case_count in zip([bars1, bars2],[Multimodal_case_count,Unimodal_case_count],[Multimodal_right_case_count,Unimodal_right_case_count]):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=24, color='black')
            ax.text(bar.get_x() + bar.get_width() / 2, height - 0.09, f'{right_case_count[i]}\n{case_count[i]}',
                    ha='center', va='bottom', fontsize=22, color='black')
            ax.hlines(y=height - 0.05, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                      linewidth=2)
    for i in range(len(x)):
        x1 = x[i] - bar_width / 2
        x2 = x[i] + bar_width / 2
        y1 = Multimodal_acc[i]
        y2 = Unimodal_acc[i]
        y_top = max(y1, y2) + 0.08
        bracket_height = 0.02
        ax.hlines(y=y_top, xmin=x1, xmax=x2, color='black', linewidth=2)
        ax.vlines(x=x1, ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
        ax.vlines(x=x2, ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
        ax.text((x1 + x2) / 2, y_top + 0.005, p_values[i], ha='center', va='bottom', fontsize=22)
    legend_handles = [
        plt.Line2D([0], [0], color=colors['Multiple'], label='Multiple', marker='s', linestyle='None', markersize=20),
        plt.Line2D([0], [0], color=colors['Single'], label='Single', marker='s', linestyle='None',markersize=20),
    ]
    ax.legend(handles=legend_handles, fontsize=28, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05), frameon=False)
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xticklabels(datasets, fontsize=28)
    ax.set_xlabel("Journal", fontsize=30, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=30, fontweight='bold')
    plt.ylim(0, 1.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig

def get_image_count_figure():
    """
    Generate a bar chart visualizing model accuracy across different image count groups,
    with case counts encoded through a continuous color scale.

    Data Source
    -----------
    Extracted by calling:
        - get_image_count_acc.image_count_acc_all_merge_data()

    Visualization
    ---------------------
    • X-axis : image count groups
    • Y-axis : Accuracy (0–1 scale)
    • Bar color : Number of cases per image count group (Purple gradient colormap)
    • Annotation :
        - Accuracy value above each bar
        - "Correct cases / Total cases" displayed inside each bar
        - Horizontal divider separating accuracy from counts text
        - Statistical brackets indicating group comparisons with p-values

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for saving or further customization.
    """
    all_result=get_image_count_acc.image_count_acc_all_merge_data('label summary.xlsx', 3, 3, 3,12,12,12)
    image_count = np.array(all_result['image_count'])
    accuracy = np.array(all_result['accuracy'])
    case_count=np.array(all_result['case_count'])
    right_case_count=np.array(all_result['right_case_count'])
    p_value = '$p$ = 0.xx'
    x = np.arange(len(image_count))
    group_start_vals = np.array([g for g in case_count])
    norm = mcolors.Normalize(vmin=group_start_vals.min(), vmax=group_start_vals.max())
    cmap = ListedColormap(cm.get_cmap('Purples')(np.linspace(0.2, 0.7, 256)))
    bar_colors = []
    for i, val in enumerate(case_count):
        bar_colors.append(cmap(norm(val)))
    fig, ax = plt.subplots(figsize=(15,10))
    bar_width = 0.6
    bars = ax.bar(x, accuracy, bar_width, color=bar_colors)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=24, color='black')
        ax.text(bar.get_x() + + bar.get_width() / 2, height - 0.09, f'{right_case_count[i]}\n{case_count[i]}',
                ha='center', va='bottom', fontsize=22, color='black')
        ax.hlines(y=height - 0.045, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                  linewidth=2)
    plt.ylim(0, 1)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(image_count, fontsize=26)
    ax.set_xlabel("Image Count", fontsize=30,fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=30,fontweight='bold')
    bracket_height = 0.02
    max_y = accuracy.max()
    x1, y1 = x[0], accuracy[0]
    x2, y2 = x[-1], accuracy[-1]
    y_top = max_y + 0.08
    ax.hlines(y=y_top, xmin=x1, xmax=x2, color='black', linewidth=2)
    ax.vlines(x=x1, ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
    ax.vlines(x=x2, ymin=y_top - bracket_height, ymax=y_top, color='black', linewidth=2)
    ax.text((x1 + x2) / 2, y_top + 0.005, p_value, ha='center', va='bottom', fontsize=22)
    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.5])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Case Count', fontsize=25)
    cbar.ax.tick_params(labelsize=17)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_icd_figure():
    """
    Generate a bar chart illustrating model accuracy across disease categories (ICD groups),
    including an additional aggregated reference bar representing overall dataset performance.

    Data Source
    -----------
    Extracted by calling:
        - get_icd_acc.icd_acc_all_data()

    Visualization
    ---------------------
    • x-axis : Disease categories mapped from ICD groups
    • Bars : Accuracy for each category
    • Color encoding :
        - Purple gradient reflects case volume per category
        - “Average” bar highlighted with gold color
    • Annotation :
        - Accuracy value displayed above each bar
        - “Correct cases / Total cases” displayed inside each bar
        - Horizontal divider marker for readability

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated ICD accuracy visualization figure for export or further styling.
    """
    all_result = get_icd_acc.icd_acc_all_data('label summary.xlsx',8,8,8,12,12,12)
    df_icd = all_result.copy()
    new_icd = "Average"
    new_accuracy = 1 # accuracy of the full dataset
    new_case_count = 1 # case count of the full dataset
    new_right_case_count = 1 # right case count of the full dataset
    new_row = pd.DataFrame({
        'icd': [new_icd],
        'accuracy': [new_accuracy],
        'case_count': [new_case_count],
        'right_case_count': [new_right_case_count]
    })
    df_icd = pd.concat([df_icd, new_row], ignore_index=True)
    df_icd = df_icd.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    green_index = df_icd[df_icd['icd'] == new_icd].index[0]
    icd = np.array(df_icd['icd'])
    accuracy = np.array(df_icd['accuracy'])
    case_count = np.array(df_icd['case_count'])
    right_case_count = np.array(df_icd['right_case_count'])
    x = np.arange(len(icd))
    norm = mcolors.Normalize(vmin=0,vmax=600)
    cmap = ListedColormap(cm.get_cmap('Purples')(np.linspace(0.2, 0.7, 256)))
    bar_colors = []
    for i, val in enumerate(case_count):
        if i == green_index:
            bar_colors.append('#f0b000')
        else:
            bar_colors.append(cmap(norm(val)))
    bar_colors[green_index] = '#f6d186'
    fig, ax = plt.subplots(figsize=(30, 8))
    bar_width = 0.8
    bars = ax.bar(x, accuracy, bar_width, color=bar_colors)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=22, color='black')
        ax.text(bar.get_x() + + bar.get_width() / 2, height - 0.11, f'{right_case_count[i]}\n{case_count[i]}',
                ha='center', va='bottom', fontsize=19, color='black')
        ax.hlines(y=height - 0.061, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                  linewidth=2)
    plt.ylim(0, 1)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(icd, fontsize=22, rotation=35, ha='right', va='top')
    ax.set_xlabel("Disease Category", fontsize=24, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=24, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_specialty_figure():
    """
    Generate a bar chart illustrating model accuracy across specialty groups,
    including an additional aggregated reference bar representing overall dataset performance.

    Data Source
    -----------
    Extracted by calling:
        - get_specialty_acc.specialty_acc_all_data()

    Visualization
    ---------------------
    • x-axis : Specialty groups
    • Bars : Accuracy for each category
    • Color encoding :
        - Purple gradient reflects case volume per category
        - “Average” bar highlighted with gold color
    • Annotation :
        - Accuracy value displayed above each bar
        - “Correct cases / Total cases” displayed inside each bar
        - Horizontal divider marker for readability

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated specialty accuracy visualization figure for export or further styling.
    """
    all_result = get_specialty_acc.specialty_acc_all_data('label summary.xlsx',9,9,9,12,12,12)
    df_specialty = all_result.copy()
    new_specialty = "Average"
    new_case_count = 1 # case count of the full dataset
    new_right_case_count = 1 # right case count of the full dataset
    new_accuracy = new_right_case_count/new_case_count
    new_row = pd.DataFrame({
        'specialty': [new_specialty],
        'accuracy': [new_accuracy],
        'case_count': [new_case_count],
        'right_case_count': [new_right_case_count]
    })
    df_specialty = pd.concat([df_specialty, new_row], ignore_index=True)
    df_specialty = df_specialty.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    green_index = df_specialty[df_specialty['specialty'] == new_specialty].index[0]
    specialty = np.array(df_specialty['specialty'])
    accuracy = np.array(df_specialty['accuracy'])
    case_count = np.array(df_specialty['case_count'])
    right_case_count = np.array(df_specialty['right_case_count'])
    x = np.arange(len(specialty))
    norm = mcolors.Normalize(vmin=0,vmax=600)
    cmap = ListedColormap(cm.get_cmap('Purples')(np.linspace(0.2, 0.7, 256)))
    bar_colors = []
    for i, val in enumerate(case_count):
        if i == green_index:
            bar_colors.append('#f0b000')
        else:
            bar_colors.append(cmap(norm(val)))
    bar_colors[green_index] = '#f6d186'
    fig, ax = plt.subplots(figsize=(30, 8))
    bar_width = 0.8
    bars = ax.bar(x, accuracy, bar_width, color=bar_colors)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=22, color='black')
        ax.text(bar.get_x() + + bar.get_width() / 2, height - 0.11, f'{right_case_count[i]}\n{case_count[i]}',
                ha='center', va='bottom', fontsize=19, color='black')
        ax.hlines(y=height - 0.06, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                  linewidth=2)
    plt.ylim(0, 1)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(specialty, fontsize=22, rotation=35, ha='right', va='top')
    ax.set_xlabel("Specialty", fontsize=24, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=24, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_image_typy_figure():
    """
    Generate a bar chart illustrating model accuracy across image modality types,
    including an additional aggregated reference bar representing overall dataset performance.

    Data Source
    -----------
    Extracted by calling:
        - get_image_type_acc.image_type_weight_acc_all_data()

    Filtering Criteria
    ------------------
    • Only include modalities with at least 30 cases
    • Exclude the "Others" modality category

    Visualization
    -------------
    • x-axis : Image modality types
    • Bars : Accuracy for each image type category
    • Color encoding :
        - Green gradient reflects case count per modality
        - "Average" bar highlighted with gold color
    • Annotation :
        - Accuracy value displayed above each bar
        - “Correct cases / Total cases” displayed inside each bar
        - Horizontal divider marker for readability

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated specialty accuracy visualization figure for export or further styling.
    """
    All_result = get_image_type_acc.image_type_weight_acc_all_data('label summary.xlsx', 4, 4, 4, 12, 12, 12)
    df_modality=All_result[(All_result['case_count'] >= 30) &
                         (All_result['image_modality'] != 'Others')]
    df_modality = df_modality.sort_values('accuracy', ascending=False)
    new_modality = "Average"
    new_accuracy = 1  # accuracy of the full dataset
    new_case_count = 1  # case count of the full dataset
    new_right_case_count = 1  # right case count of the full dataset
    new_row = pd.DataFrame({
        'image_modality': [new_modality],
        'accuracy': [new_accuracy],
        'case_count': [new_case_count],
        'right_case_count': [new_right_case_count]
    })
    df_modality = pd.concat([df_modality, new_row], ignore_index=True)
    df_modality = df_modality.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    green_index = df_modality[df_modality['image_modality'] == new_modality].index[0]
    image_modality = np.array(df_modality['image_modality'])
    accuracy = np.array(df_modality['accuracy'])
    case_count = np.array(df_modality['case_count'])
    right_case_count = np.array(df_modality['right_case_count'])
    x = np.arange(len(image_modality))
    group_start_vals = np.array([g for g in case_count])
    norm = mcolors.Normalize(vmin=0, vmax=1500)
    cmap = ListedColormap(cm.get_cmap('Greens')(np.linspace(0.2, 0.7, 256)))
    bar_colors = []
    for i, val in enumerate(case_count):
        if i == green_index:
            bar_colors.append('#f6d186')
        else:
            bar_colors.append(cmap(norm(val)))
    bar_colors[green_index] = '#f6d186'
    fig, ax = plt.subplots(figsize=(27, 8))
    bar_width = 0.8
    bars = ax.bar(x, accuracy, bar_width, color=bar_colors)
    for i,bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + + bar.get_width() / 2, height + 0.005, f'{height:.2f}',
                ha='center', va='bottom', fontsize=26, color='black')
        ax.text(bar.get_x() + + bar.get_width() / 2, height - 0.125, f'{right_case_count[i]:.1f}\n{case_count[i]:.1f}',
                ha='center', va='bottom', fontsize=24, color='black')
        ax.hlines(y=height - 0.064, xmin=bar.get_x() + 0.03, xmax=bar.get_x() + bar_width - 0.03, color='black',
                  linewidth=2)
    plt.ylim(0, 1)
    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.5])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 必需项
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Case Count', fontsize=26)
    cbar.ax.tick_params(labelsize=22)
    ax.tick_params(axis='y', labelsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(image_modality, fontsize=28,rotation=35, ha='right', va='top')
    ax.set_xlabel("Image Type", fontsize=30, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=30, fontweight='bold')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def get_image_type_region_figure():
    """
    Generate a multi-panel figure showing model accuracy across anatomical regions
    for different medical imaging types, with each image type displayed in a separate subplot.

    Data Source
    -----------
    Extracted by calling:
        - get_region_acc.image_type_region_weighted_acc_all_data()

    Filtering Criteria
    ------------------
    • Only include image types that have data for more than one anatomical region
    • Exclude modalities with data for only one region (insufficient for comparison)

    Visualization
    -------------
    • Multi-panel Layout: Subplots arranged in a grid (2 columns, variable rows)
    • Each Subplot:
        - x-axis: Anatomical regions (standardized list: Head/neck, Thorax, Abdomen, etc.)
        - y-axis: Accuracy values (0-1 scale)
        - Bars: Colored by region using a consistent "Paired" color palette
        - Title: Image type name
    • Bar Annotations:
        - Accuracy value displayed above each bar (2 decimal places)
        - "right_case_count / total_case_count" displayed inside each bar
        - Horizontal divider line inside each bar for visual separation
    • Layout Features:
        - Shared x-axis label: "Anatomical Region"
        - Shared y-axis label: "Accuracy"
        - Empty subplots are removed when number of modalities < grid capacity

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The generated multi-panel visualization figure with modality-specific region accuracy plots.
        Returns None if no eligible modalities are found for plotting.
    """
    all_result = get_region_acc.image_type_region_weighted_acc_all_data('label summary.xlsx', 4,4,4, 10, 10, 10,12, 12, 12)
    df_region = all_result
    modalities = [
        m for m in df_region['image_modality'].unique()
        if df_region[df_region['image_modality'] == m]['region'].nunique() > 1
    ]
    if len(modalities) == 0:
        print("No eligible modalities (region count > 1) found for plotting.")
        return None
    cols = 2
    rows = (len(modalities) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 12, rows * 4.5))
    axes = axes.flatten()
    all_regions = ["Head/neck", "Thorax", "Abdomen", "Pelv/perin.", "Upper limb", "Lower limb",  "Back", "Others"]
    palette = sns.color_palette("Paired", len(all_regions))
    region_color_map = {region: palette[i] for i, region in enumerate(all_regions)}
    bar_width = 0.8
    for idx, imgmodality in enumerate(modalities):
        sub_df = df_region[df_region['image_modality'] == imgmodality]
        ax = axes[idx]
        sub_df = sub_df[sub_df['region'].isin(all_regions)]
        sub_df['region'] = pd.Categorical(sub_df['region'], categories=all_regions, ordered=True)
        sub_df = sub_df.sort_values('region')
        regions = sub_df['region']
        accuracy = sub_df['accuracy']
        case_count = sub_df['case_count']
        right_case_count = sub_df['right_case_count'].round(1)
        if len(regions) == 0:
            fig.delaxes(ax)
            continue
        bar_colors = [region_color_map[r] for r in regions]
        bars = ax.bar(regions, accuracy, width=bar_width, color=bar_colors)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, fontsize=20, rotation=18)
        ax.set_title(imgmodality, fontsize=22, pad=-5, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=18)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=20, color='black')
            ax.text(x, height - 0.28, f'{right_case_count.iloc[i]:.1f}\n{case_count.iloc[i]:.1f}',
                    ha='center', va='bottom', fontsize=20, color='black')
            ax.hlines(y=height - 0.16, xmin=bar.get_x() + 0.08, xmax=bar.get_x() + bar_width - 0.08,
                      color='black', linewidth=1.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    for i in range(len(modalities), len(axes)):
        fig.delaxes(axes[i])
    fig.text(0.56, 0.08, 'Anatomical Region', ha='center', fontsize=26, fontweight='bold')
    fig.text(0.08, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=26, fontweight='bold')
    plt.tight_layout(rect=[0.1, 0.1, 1, 1])
    plt.show()
    return fig

def get_word_count_figure():
    """
    Generate a scatter plot with trend lines showing the relationship between
    text word count and model accuracy across different medical journals.

    Data Sources
    ------------
    Loaded by calling:
        - get_word_count_acc.word_count_acc_data() for individual journals (Lancet, NEJM, JAMA)
        - get_word_count_acc.word_count_acc_all_data() for aggregated data across all journals

    Processing
    ----------
    • Word count extraction: Convert bin labels (e.g., "0-49") to midpoint values (25)
    • Marker sizing: Point sizes proportional to case count in each bin (×5 scaling factor)
    • Visualization elements:
        - Dashed lines connecting accuracy points for each journal
        - Semi-transparent scatter points sized by case volume
        - Linear regression line for aggregated data
        - Custom color scheme for journal differentiation

    Visualization Components
    ------------------------
    • Lancet: Green series
    • NEJM: Orange series
    • JAMA: Blue series
    • All Journals: Red series with regression line
    • x-axis: Word count bin midpoints with original range labels
    • y-axis: Accuracy values (0-1 scale)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object for further saving or customization.
    """
    Lancet_result = get_word_count_acc.word_count_acc_data('Lancet_json_record.json', 'label summary.xlsx', 0, 12)
    NEJM_result = get_word_count_acc.word_count_acc_data('NEJM_json_record.json', 'label summary.xlsx', 1, 12)
    JAMA_result = get_word_count_acc.word_count_acc_data('JAMA_json_record.json', 'label summary.xlsx', 2, 12)
    all_result = get_word_count_acc.word_count_acc_all_data('Lancet_json_record.json', 'NEJM_json_record.json', 'JAMA_json_record.json',
                                                            'label summary.xlsx',
                                                            0, 1, 2,
                                                            12, 12, 12)
    word_count = np.array(all_result['word_count_group'].str.extract(r'(\d+)')[0].astype(int)+25)
    accuracy = np.array(all_result['accuracy'])
    case_count = np.array(all_result['case_count'])
    word_count_Lancet = np.array(Lancet_result['word_count_group'].str.extract(r'(\d+)')[0].astype(int) + 25)
    accuracy_Lancet = np.array(Lancet_result['accuracy'])
    case_count_Lancet = np.array(Lancet_result['case_count'])
    word_count_NEJM = np.array(NEJM_result['word_count_group'].str.extract(r'(\d+)')[0].astype(int) + 25)
    accuracy_NEJM = np.array(NEJM_result['accuracy'])
    case_count_NEJM = np.array(NEJM_result['case_count'])
    word_count_JAMA = np.array(JAMA_result['word_count_group'].str.extract(r'(\d+)')[0].astype(int) + 25)
    accuracy_JAMA = np.array(JAMA_result['accuracy'])
    case_count_JAMA = np.array(JAMA_result['case_count'])
    fig, ax = plt.subplots(figsize=(18,10))
    plt.plot(word_count_Lancet, accuracy_Lancet, marker='.', color='#6A994E',linestyle='--', linewidth=2,zorder=1)
    plt.scatter(word_count_Lancet, accuracy_Lancet, s=case_count_Lancet * 5, color='#6A994E', alpha=0.5,zorder=1)
    plt.plot(word_count_NEJM, accuracy_NEJM, marker='.', color='#FFA94D', linestyle='--', linewidth=2,zorder=2)
    plt.scatter(word_count_NEJM, accuracy_NEJM, s=case_count_NEJM * 5, color='#FFA94D', alpha=0.5,zorder=2)
    plt.plot(word_count_JAMA, accuracy_JAMA, marker='.', color='#355C7D', linestyle='--', linewidth=2,zorder=3)
    plt.scatter(word_count_JAMA, accuracy_JAMA, s=case_count_JAMA * 5, color='#355C7D', alpha=0.5,zorder=3)
    sns.scatterplot(x=word_count, y=accuracy, color='#BC4749',zorder=4)
    plt.scatter(word_count, accuracy, s=case_count * 5, color='#BC4749', alpha=0.5,zorder=4)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    sns.regplot(x=word_count, y=accuracy, scatter=False, line_kws={'color': '#D90368'})
    ax.set_xlabel("Word Count", fontsize=30, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=30, fontweight='bold')
    xticks = word_count-25
    xtick_labels = [str(x-25) if x < 400 else '400+' for x in word_count]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    plt.ylim(0, 1.05)
    for label, color in zip([ 'Lancet ($p$ = 0.xx)', 'NEJM ($p$ = 0.xx)', 'JAMA ($p$ = 0.xx)','All ($p$ = 0.xx)'],
                            ['#6A994E', '#FFA94D', '#355C7D','#BC4749']):
        ax.scatter([], [], color=color, label=label, s=100)
    ax.legend(fontsize=27,frameon=False, loc='lower right')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    return fig

def main():
    overall_acc_figure=get_overall_acc_figure()
    date_figure=get_date_figure()
    sex_figure=get_sex_figure()
    age_figure=get_age_figure()
    task_type_figure=get_task_type_figure()
    image_type_count_figure=get_image_type_count_figure()
    image_count_figure=get_image_count_figure()
    icd_figure=get_icd_figure()
    specialty_figure=get_specialty_figure()
    image_typy_figure=get_image_typy_figure()
    image_type_region_figure=get_image_type_region_figure()
    word_count_figure=get_word_count_figure()


if __name__ == "__main__":
    main()