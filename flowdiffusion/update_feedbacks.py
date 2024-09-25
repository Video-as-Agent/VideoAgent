import os

def update_feedback_file(feedback_file_path):
    # Read the existing lines from the feedback file
    with open(feedback_file_path, 'r') as file:
        lines = file.readlines()
    
    # Check if there are at least two lines to update
    if len(lines) >= 2:
        lines[0] = "base model: " + lines[0]
        lines[1] = "Iteration_1_model_304: " + lines[1]
        
        # Write the updated lines back to the feedback file
        with open(feedback_file_path, 'w') as file:
            file.writelines(lines)
            
def delete_third_line(feedback_file_path):
    # Read the existing lines from the feedback file
    with open(feedback_file_path, 'r') as file:
        lines = file.readlines()
    
    # Retain only the first two lines
    lines = lines[:2]
    
    # Write the updated lines back to the feedback file
    with open(feedback_file_path, 'w') as file:
        file.writelines(lines)
        
def delete_empty_lines(feedback_file_path):
    # Read the existing lines from the feedback file
    with open(feedback_file_path, 'r') as file:
        lines = file.readlines()
    
    # Remove any empty lines
    lines = [line for line in lines if line.strip()]

    # Write the updated lines back to the feedback file
    with open(feedback_file_path, 'w') as file:
        file.writelines(lines)

def update_all_feedback_files(base_path):
    for root, dirs, files in os.walk(base_path):
        for dir in dirs:
            if "trajectory" in dir:
                trajectory_dir = os.path.join(root, dir)
                feedback_file_path = os.path.join(trajectory_dir, 'feedback.txt')
                
                if os.path.exists(feedback_file_path):
                    delete_empty_lines(feedback_file_path)
                    print(f"Updated {feedback_file_path}")

def main():
    base_path = '/home/ubuntu/achint/datasets/metaworld_split/metaworld_dataset'
    update_all_feedback_files(base_path)

if __name__ == "__main__":
    main()
