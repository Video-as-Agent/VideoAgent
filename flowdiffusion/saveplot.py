import pandas as pd
import matplotlib.pyplot as plt

def plot_accept_percentages_per_task(csv_file, title):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract accept percentages
    iterations = ['base_model', 'Iteration_1', 'Iteration_2', 'Iteration_305_310_315']
    accept_columns = [f'{iter}_Accept%' for iter in iterations]
    
    # Convert percentage strings to floats
    for col in accept_columns:
        df[col] = df[col].str.rstrip('%').astype('float')
    
    # Plot
    plt.figure(figsize=(12, 8))
    for _, row in df.iterrows():
        plt.plot(iterations, row[accept_columns], marker='o', label=row['Task'])
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Accept Percentage')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot for max voting
plot_accept_percentages_per_task('/home/ubuntu/achint/SI-GenSim/flowdiffusion/final_feedback_percentages.csv', 'Accept Percentages Per Task - Out of 5 feedbacks')

# Plot for any accept
plot_accept_percentages_per_task('/home/ubuntu/achint/SI-GenSim/flowdiffusion/final_feedback_percentages_any_accept.csv', 'Accept Percentages Per Task')

print("Graphs have been generated and saved as PNG files.")