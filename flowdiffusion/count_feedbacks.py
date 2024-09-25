# import os
# import csv
# from collections import defaultdict

# def clean_feedback(feedback):
#     return feedback.strip().rstrip('.').lower()

# def compile_feedback(base_path):
#     feedback_counts = defaultdict(lambda: defaultdict(lambda: {'accept': 0, 'reject': 0, 'total': 0}))

#     for root, dirs, files in os.walk(base_path):
#         relevant_files = ['feedback.txt', 'feedback_1.txt', 'feedback_2.txt', 'feedback_305_310_315.txt']
#         if any(file in files for file in relevant_files):
#             path_parts = root.split(os.sep)
#             task_index = path_parts.index('metaworld_dataset') + 1 if 'metaworld_dataset' in path_parts else -1
#             task = path_parts[task_index] if task_index < len(path_parts) and task_index != -1 else "unknown"

#             for file in relevant_files:
#                 if file in files:
#                     with open(os.path.join(root, file), 'r') as f:
#                         lines = f.readlines()
#                         if file == 'feedback.txt':
#                             if lines and lines[0].startswith('base model:'):
#                                 feedback = clean_feedback(lines[0].split(',')[1])
#                                 feedback_counts[task]['base_model'][feedback] += 1
#                                 feedback_counts[task]['base_model']['total'] += 1
#                         else:
#                             accepts = sum('accept' in clean_feedback(line) for line in lines)
#                             rejects = sum('reject' in clean_feedback(line) for line in lines)
#                             key = file.replace('.txt', '').replace('feedback_', 'Iteration_')
#                             feedback_counts[task][key]['accept'] += accepts
#                             feedback_counts[task][key]['reject'] += rejects
#                             feedback_counts[task][key]['total'] += accepts + rejects

#     # Calculate percentages and write to CSV
#     output_file = 'final_feedback_percentages.csv'
#     with open(output_file, 'w', newline='') as csvfile:
#         fieldnames = ['Task', 'base_model_Accept%', 'base_model_Reject%', 
#                       'Iteration_1_Accept%', 'Iteration_1_Reject%', 
#                       'Iteration_2_Accept%', 'Iteration_2_Reject%', 
#                       'Iteration_305_310_315_Accept%', 'Iteration_305_310_315_Reject%']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         writer.writeheader()
#         for task, data in feedback_counts.items():
#             row = {'Task': task}
#             for feedback_type, counts in data.items():
#                 total = counts['total']
#                 if total > 0:
#                     accept_percent = (counts['accept'] / total) * 100
#                     reject_percent = (counts['reject'] / total) * 100
#                     row[f'{feedback_type}_Accept%'] = f'{accept_percent:.2f}%'
#                     row[f'{feedback_type}_Reject%'] = f'{reject_percent:.2f}%'
#                 else:
#                     row[f'{feedback_type}_Accept%'] = 'N/A'
#                     row[f'{feedback_type}_Reject%'] = 'N/A'
#             writer.writerow(row)

#     print(f"Final feedback percentages CSV has been created: {output_file}")

# # Usage
# base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset/'

# compile_feedback(base_path)

import os
import csv
from collections import defaultdict

def clean_feedback(feedback):
    return feedback.strip().rstrip('.').lower()

def compile_feedback_any_accept(base_path):
    feedback_counts = defaultdict(lambda: defaultdict(lambda: {'accept': 0, 'reject': 0, 'total': 0}))

    for root, dirs, files in os.walk(base_path):
        relevant_files = ['feedback.txt', 'feedback_1.txt', 'feedback_2.txt', 'feedback_305_310_315.txt']
        if any(file in files for file in relevant_files):
            path_parts = root.split(os.sep)
            task_index = path_parts.index('metaworld_dataset') + 1 if 'metaworld_dataset' in path_parts else -1
            task = path_parts[task_index] if task_index < len(path_parts) and task_index != -1 else "unknown"

            for file in relevant_files:
                if file in files:
                    with open(os.path.join(root, file), 'r') as f:
                        lines = f.readlines()
                        if file == 'feedback.txt':
                            if lines and lines[0].startswith('base model:'):
                                feedback = clean_feedback(lines[0].split(',')[1])
                                feedback_counts[task]['base_model'][feedback] += 1
                                feedback_counts[task]['base_model']['total'] += 1
                        else:
                            # Implement "any accept" policy
                            has_accept = any('accept' in clean_feedback(line) for line in lines)
                            key = file.replace('.txt', '').replace('feedback_', 'Iteration_')
                            if has_accept:
                                feedback_counts[task][key]['accept'] += 1
                            else:
                                feedback_counts[task][key]['reject'] += 1
                            feedback_counts[task][key]['total'] += 1

    # Calculate percentages and write to CSV
    output_file = 'final_feedback_percentages_any_accept.csv'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Task', 'base_model_Accept%', 'base_model_Reject%', 
                      'Iteration_1_Accept%', 'Iteration_1_Reject%', 
                      'Iteration_2_Accept%', 'Iteration_2_Reject%', 
                      'Iteration_305_310_315_Accept%', 'Iteration_305_310_315_Reject%']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for task, data in feedback_counts.items():
            row = {'Task': task}
            for feedback_type, counts in data.items():
                total = counts['total']
                if total > 0:
                    accept_percent = (counts['accept'] / total) * 100
                    reject_percent = (counts['reject'] / total) * 100
                    row[f'{feedback_type}_Accept%'] = f'{accept_percent:.2f}%'
                    row[f'{feedback_type}_Reject%'] = f'{reject_percent:.2f}%'
                else:
                    row[f'{feedback_type}_Accept%'] = 'N/A'
                    row[f'{feedback_type}_Reject%'] = 'N/A'
            writer.writerow(row)

    print(f"Final feedback percentages CSV (Any Accept policy) has been created: {output_file}")

# Usage
base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset/'

compile_feedback_any_accept(base_path)