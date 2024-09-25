# import os

# def create_feedback_train_files(base_path):
#     files_created = 0

#     for root, dirs, files in os.walk(base_path):
#         if 'feedback.txt' in files:
#             feedback_file_path = os.path.join(root, 'feedback.txt')
#             feedback_train_file_path = os.path.join(root, 'feedback_train.txt')
            
#             with open(feedback_file_path, 'r') as f_in:
#                 first_line = f_in.readline().strip()
                
#                 if first_line.startswith('base model:'):
#                     with open(feedback_train_file_path, 'w') as f_out:
#                         f_out.write(first_line + '\n')
#                     files_created += 1

#     print(f"Created {files_created} feedback_train.txt files.")

# # Usage
# base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset/'

# create_feedback_train_files(base_path)

import os

def update_feedback_train_files(base_path):
    files_updated = 0

    for root, dirs, files in os.walk(base_path):
        if 'feedback_train.txt' in files and 'feedback_1.txt' in files:
            feedback_train_file_path = os.path.join(root, 'feedback_train.txt')
            feedback_1_file_path = os.path.join(root, 'feedback_1.txt')
            
            # Get task name from the path
            path_parts = root.split(os.sep)
            task_index = path_parts.index('metaworld_dataset') + 1 if 'metaworld_dataset' in path_parts else -1
            task = path_parts[task_index] if task_index < len(path_parts) and task_index != -1 else "unknown"

            # Analyze feedback_1.txt
            with open(feedback_1_file_path, 'r') as f_in:
                lines = f_in.readlines()
                accepts = sum('Accept' in line for line in lines)
                rejects = sum('Reject' in line for line in lines)
            
            majority_vote = 'Accept' if accepts > rejects else 'Reject'
            iteration_line = f"Iteration_1_model_305:{task}, {majority_vote}"
            
            # Update feedback_train.txt
            with open(feedback_train_file_path, 'r+') as f:
                content = f.read()
                if iteration_line not in content:  # Avoid duplicate entries
                    f.seek(0, 2)  # Move to the end of the file
                    f.write(iteration_line + '\n')
                    files_updated += 1

    print(f"Updated {files_updated} feedback_train.txt files.")

# Usage
base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset/'

update_feedback_train_files(base_path)