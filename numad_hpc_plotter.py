import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/egor/Downloads/HPC_numad/numad_benchmark_results.csv')
static_df = df[df['Mode'] == 'Static_NoNumad'].reset_index()
opt_df = df[df['Mode'] == 'Bayesian_Optimized'].reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Plot 1: Hit Rates vs Misses Comparison
axes[0, 0].plot(static_df['Cycle'], static_df['HitRate'], label='Static (No Tuning)', color='red', linestyle='--')
axes[0, 0].plot(opt_df['Cycle'], opt_df['HitRate'], label='Bayesian Optimized', color='green')
axes[0, 0].set_title('NUMA Node Allocations (Hit Rate %)')
axes[0, 0].legend()

# Plot 2: Uncontrolled Page Migrations
axes[0, 1].plot(static_df['Cycle'], static_df['Migrated'], label='Static (No Tuning)', color='red', linestyle='--')
axes[0, 1].plot(opt_df['Cycle'], opt_df['Migrated'], label='Bayesian Optimized', color='green')
axes[0, 1].set_title('Page Migrations Over Time')
axes[0, 1].legend()

# Plot 3: System Memory Stall Pressures
axes[1, 0].plot(static_df['Cycle'], static_df['MemPress'], label='Static', color='red', linestyle='--')
axes[1, 0].plot(opt_df['Cycle'], opt_df['MemPress'], label='Bayesian Optimized', color='green')
axes[1, 0].set_title('Memory Pressure Stall Metrics')
axes[1, 0].legend()

# Plot 4: Dynamic Parameter Adaptations Tracking (-u, -m)
axes[1, 1].plot(opt_df['Cycle'], opt_df['Param_u'], label='Target Utilization (-u)', color='blue')
axes[1, 1].plot(opt_df['Cycle'], opt_df['Param_m'], label='Memory Locality (-m)', color='purple')
axes[1, 1].set_title('Dynamic Parameter Path Convergence')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/Users/egor/Downloads/HPC_numad/numad_performance_trends.png')
plt.show()