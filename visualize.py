import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from constants import VIZ_DIR, OUTPUT_DIR
from scipy import stats


def parse_json_files(root_folder):
    """
    Recursively traverse the folder structure starting at root_folder,
    and extract the 'sol_exec_time' value from each JSON file.
    
    Assumes that:
      - Each global step is a folder under root_folder.
      - Each global step folder contains 192 problem folders.
      - Each problem folder contains 8 JSON files.
    
    Returns:
        A dictionary where keys are problem_ids and values are lists of sol_exec_times.
    """
    problem_exec_times = defaultdict(list)

    # Walk through all subdirectories and files under root_folder.
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Check if the key 'sol_exec_time' exists in the JSON.
                        if 'sol_exec_time' in data:
                            # Extract problem_id from the directory path
                            # Assuming the directory structure is: root_folder/problem_id/...
                            path_parts = os.path.normpath(dirpath).split(os.sep)
                            # Find the problem_id in the path
                            # It should be the directory name directly under the global step directory
                            relative_path = os.path.relpath(dirpath, root_folder)
                            problem_parts = relative_path.split(os.sep)
                            if len(problem_parts) > 0:
                                problem_id = problem_parts[0]  # First directory under global step
                                problem_exec_times[problem_id].append(data['sol_exec_time'])
                            else:
                                print(f"Could not determine problem_id for {file_path}")
                        else:
                            print(f"'sol_exec_time' not found in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return dict(problem_exec_times)


def get_global_step_times(experiment_path):
    """
    Get sol_exec_time for each problem_id for each global step in the experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary mapping global step (as int) to dictionaries of problem_id -> sol_exec_times
    """
    global_step_times = {}
    
    # List all directories in the experiment path (these should be global steps)
    try:
        global_step_dirs = [d for d in os.listdir(experiment_path) 
                           if os.path.isdir(os.path.join(experiment_path, d))]
    except FileNotFoundError:
        print(f"Experiment directory not found: {experiment_path}")
        return {}
    
    for step_dir in global_step_dirs:
        try:
            # Extract the global step number from the directory name
            # Format is expected to be "global_step_X"
            if step_dir.startswith("global_step_"):
                global_step = int(step_dir.split("_")[-1])
                step_path = os.path.join(experiment_path, step_dir)
                
                # Get all sol_exec_times for this global step, organized by problem_id
                problem_times = parse_json_files(step_path)
                if problem_times:
                    global_step_times[global_step] = problem_times
                else:
                    print(f"No sol_exec_time values found in global step {global_step}")
            else:
                print(f"Skipping directory {step_dir} - not in expected format 'global_step_X'")
        except (ValueError, IndexError) as e:
            # Skip directories that don't match the expected format
            print(f"Skipping directory {step_dir} - could not extract step number: {e}")
    
    return global_step_times


def detect_outliers(data, method='iqr', threshold=1.5):
    """
    Detect outliers in a dataset using various methods.
    
    Args:
        data: List or array of values
        method: Method to use for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (filtered_data, outliers, is_outlier_mask)
    """
    data = np.array(data)
    is_outlier = np.zeros(len(data), dtype=bool)
    
    if method == 'iqr':
        # IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        is_outlier = (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        # Z-score method
        z_scores = stats.zscore(data)
        is_outlier = np.abs(z_scores) > threshold
    
    filtered_data = data[~is_outlier]
    outliers = data[is_outlier]
    
    return filtered_data, outliers, is_outlier


def plot_step_differences(global_step_times, experiment_name):
    """
    Create histograms of the differences in average sol_exec_time between adjacent global steps.
    
    Args:
        global_step_times: Dictionary mapping global step to dictionaries of problem_id -> sol_exec_times
        experiment_name: Name of the experiment for plot titles and filenames
    """
    if not global_step_times:
        print("No data to plot")
        return
    
    # Sort global steps
    sorted_steps = sorted(global_step_times.keys())
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(VIZ_DIR, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average sol_exec_time for each problem in each global step
    # Use median instead of mean for robustness against outliers
    avg_times_by_problem = {}
    for step, problem_times in global_step_times.items():
        avg_times_by_problem[step] = {}
        for problem_id, times in problem_times.items():
            if times:  # Only calculate if we have times for this problem
                # Filter out outliers for more robust average calculation
                filtered_times, _, _ = detect_outliers(times, method='iqr', threshold=2.0)
                if len(filtered_times) > 0:
                    # Use median for robustness
                    avg_times_by_problem[step][problem_id] = np.median(filtered_times)
                else:
                    # If all values were outliers, use the median of the original data
                    avg_times_by_problem[step][problem_id] = np.median(times)
    
    # Calculate overall average for each step (for the trend plot)
    avg_times_by_step = {}
    for step, problem_avgs in avg_times_by_problem.items():
        if problem_avgs:
            # Use median of medians for overall robustness
            avg_times_by_step[step] = np.median(list(problem_avgs.values()))
        else:
            avg_times_by_step[step] = 0
    
    # Plot histogram of differences between adjacent global steps
    for i in range(len(sorted_steps) - 1):
        current_step = sorted_steps[i]
        next_step = sorted_steps[i+1]
        
        # Get the average execution times for both steps
        current_avgs = avg_times_by_problem[current_step]
        next_avgs = avg_times_by_problem[next_step]
        
        # Find common problem IDs between the two steps
        common_problems = set(current_avgs.keys()) & set(next_avgs.keys())
        
        if not common_problems:
            print(f"No common problems found between steps {current_step} and {next_step}")
            continue
        
        # Calculate differences in average execution time for each common problem
        problem_diffs = []
        problem_ids = []  # Keep track of problem IDs for outlier analysis
        for problem_id in common_problems:
            current_avg = current_avgs[problem_id]
            next_avg = next_avgs[problem_id]
            problem_diffs.append(next_avg - current_avg)
            problem_ids.append(problem_id)
        
        # Filter out outliers for more robust visualization
        filtered_diffs, outlier_diffs, is_outlier = detect_outliers(problem_diffs, method='iqr', threshold=2.0)
        
        # Create histogram of the differences in average execution times
        plt.figure(figsize=(10, 6))
        
        # Plot histogram with filtered data
        n, bins, patches = plt.hist(filtered_diffs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add a vertical line at 0
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # Add plot details
        plt.title(f'Difference in Average sol_exec_time by Problem: Step {next_step} - Step {current_step}')
        plt.xlabel('Difference in Average Execution Time (seconds)')
        plt.ylabel('Number of Problems')
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot (using robust statistics)
        avg_diff = np.median(problem_diffs)  # Use median instead of mean
        iqr_diff = np.percentile(problem_diffs, 75) - np.percentile(problem_diffs, 25)
        
        stats_text = (
            f'Median Difference: {avg_diff:.4f}s\n'
            f'IQR: {iqr_diff:.4f}s\n'
            f'Outliers removed: {sum(is_outlier)} / {len(problem_diffs)}'
        )
        
        plt.annotate(stats_text, xy=(0.7, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save the plot
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'diff_step_{current_step}_to_{next_step}.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved difference histogram to {output_file}")
        
        # If there are outliers, create a separate plot to show them
        if sum(is_outlier) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(outlier_diffs)), outlier_diffs, color='orange')
            plt.title(f'Outlier Differences: Step {next_step} - Step {current_step}')
            plt.xlabel('Problem Index')
            plt.ylabel('Difference in Execution Time (seconds)')
            plt.grid(True, alpha=0.3)
            
            # Add problem IDs for the outliers
            outlier_ids = [problem_ids[i] for i in range(len(problem_ids)) if is_outlier[i]]
            plt.xticks(range(len(outlier_diffs)), outlier_ids, rotation=90)
            
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'outliers_step_{current_step}_to_{next_step}.png')
            plt.savefig(output_file)
            plt.close()
            
            print(f"Saved outlier plot to {output_file}")
        
        # Create a box plot for a more robust visualization
        plt.figure(figsize=(8, 6))
        plt.boxplot(problem_diffs, vert=True, patch_artist=True, showfliers=False)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
        plt.title(f'Box Plot of Differences: Step {next_step} - Step {current_step}')
        plt.ylabel('Difference in Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'boxplot_step_{current_step}_to_{next_step}.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved box plot to {output_file}")
        
        # Also create a scatter plot comparing average times for each problem
        plt.figure(figsize=(10, 8))
        
        # Extract the average times for common problems
        current_problem_avgs = [current_avgs[pid] for pid in common_problems]
        next_problem_avgs = [next_avgs[pid] for pid in common_problems]
        
        # Filter out outliers for scatter plot
        non_outlier_indices = np.where(~is_outlier)[0]
        current_filtered = [current_problem_avgs[i] for i in non_outlier_indices]
        next_filtered = [next_problem_avgs[i] for i in non_outlier_indices]
        
        # Determine the axis limits (excluding outliers)
        if current_filtered and next_filtered:
            max_val = max(max(current_filtered), max(next_filtered)) * 1.1
        else:
            max_val = 1.0  # Default if no data
        
        # Plot the scatter plot (excluding outliers)
        plt.scatter(current_filtered, next_filtered, alpha=0.6, label='Regular Points')
        
        # Plot outliers with different color and marker
        if sum(is_outlier) > 0:
            outlier_indices = np.where(is_outlier)[0]
            current_outliers = [current_problem_avgs[i] for i in outlier_indices]
            next_outliers = [next_problem_avgs[i] for i in outlier_indices]
            plt.scatter(current_outliers, next_outliers, color='red', marker='x', alpha=0.7, label='Outliers')
        
        # Add a diagonal line (y=x) to show where points would fall if there was no change
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        # Add plot details
        plt.title(f'Average sol_exec_time Comparison: Step {current_step} vs Step {next_step}')
        plt.xlabel(f'Step {current_step} Average Execution Time (seconds)')
        plt.ylabel(f'Step {next_step} Average Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Make the plot square and set equal axis limits
        plt.axis('equal')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        
        # Add statistics to the plot
        stats_text = (
            f'Number of problems: {len(common_problems)}\n'
            f'Median difference: {avg_diff:.4f}s\n'
            f'Problems faster in step {next_step}: {sum(d < 0 for d in problem_diffs)}\n'
            f'Problems slower in step {next_step}: {sum(d > 0 for d in problem_diffs)}'
        )
        
        plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     va='top')
        
        # Save the plot
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'scatter_step_{current_step}_to_{next_step}.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Saved scatter plot to {output_file}")
    
    # Create a summary plot showing the trend of average execution times
    plt.figure(figsize=(12, 6))
    steps = sorted_steps
    avgs = [avg_times_by_step[step] for step in steps]
    
    plt.plot(steps, avgs, 'o-', linewidth=2, markersize=8)
    plt.title(f'Median sol_exec_time Trend - {experiment_name}')
    plt.xlabel('Global Step')
    plt.ylabel('Median Execution Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Save the trend plot
    trend_file = os.path.join(output_dir, 'avg_exec_time_trend.png')
    plt.tight_layout()
    plt.savefig(trend_file)
    plt.close()
    
    print(f"Saved trend plot to {trend_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize differences in sol_exec_time between global steps')
    parser.add_argument('--experiment_name', type=str, required=True, 
                        help='Name of the experiment to analyze')
    
    args = parser.parse_args()
    
    # Construct the path to the experiment directory
    experiment_path = os.path.join(OUTPUT_DIR, args.experiment_name)
    
    print(f"Analyzing experiment: {args.experiment_name}")
    print(f"Looking for data in: {experiment_path}")
    
    # Get execution times for each global step
    global_step_times = get_global_step_times(experiment_path)
    
    if not global_step_times:
        print(f"No data found for experiment {args.experiment_name}")
        return
    
    # Plot the differences between adjacent global steps
    plot_step_differences(global_step_times, args.experiment_name)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()