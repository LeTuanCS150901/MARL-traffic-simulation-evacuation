import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, for prettier styling
import os

# --- 1. CONFIGURATION (Change these values to update your plot) ---
FILE_PATH     = 'gradient_stats.txt'           # Your filename
SELECT_GROUP  = '-60995980'                  # The group id
SELECT_LAYER  = 'hidden_layers.1.weight'          # The layer name

# SELECT_LAYER = 'transformer.layers.0.self_attn.in_proj_weight'
SELECT_METRIC = ['mean', 'norm']               # Choose: ['mean', 'std', 'norm']

def main():
    # --- 2. LOAD & PREPARE DATA ---
    if os.path.exists(FILE_PATH):
        df = pd.read_csv(FILE_PATH)
        df.columns = [c.strip() for c in df.columns]  # Clean whitespace
        
        # Filter the data
        mask = (df['group'] == SELECT_GROUP) & (df['layer'] == SELECT_LAYER)
        subset = df[mask].sort_values('step')

        if subset.empty:
            print(f"⚠️ No data found for Group: {SELECT_GROUP} | Layer: {SELECT_LAYER}")
            if 'layer' in df.columns:
                print(f"Available Layers: {df['layer'].unique()[:5]}...") 
        else:
            # --- 3. PLOTTING ---
            sns.set_style("whitegrid")  # Sets a clean background
            fig, ax1 = plt.subplots(figsize=(12, 6))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(SELECT_METRIC):
                ax1.plot(subset['step'], subset[metric], 
                        label=metric.upper(), 
                        marker='o', 
                        linewidth=2, 
                        color=colors[i % len(colors)])

            # Formatting
            plt.title(f"Training Metrics Over Time\nLayer: {SELECT_LAYER}", fontsize=14, pad=20)
            ax1.set_xlabel("Step", fontsize=12)
            ax1.set_ylabel("Value", fontsize=12)
            ax1.legend(loc='upper left', frameon=True)
            
            plt.tight_layout()
            
            # Save the plot (optional)
            # plt.savefig('training_metrics.png')
            # print("✅ Plot saved as training_metrics.png")
            
            plt.show()
    else:
        print(f"❌ Error: File '{FILE_PATH}' not found in the current directory.")

if __name__ == "__main__":
    main()