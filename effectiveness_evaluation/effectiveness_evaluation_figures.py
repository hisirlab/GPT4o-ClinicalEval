import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from PIL import Image, ImageDraw, ImageFont
from matplotlib import image as mpimg, cm
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error

import get_dataset_acc
import get_date_acc


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

def main():
    overall_acc_figure=get_overall_acc_figure()
    date_figure=get_date_figure()

if __name__ == "__main__":
    main()