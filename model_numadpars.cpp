#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <iomanip>

// Input metrics that indicates the numad performance in the system
struct SystemMetrics {
    double numad_hit;
    double numa_miss;
    double numa_other;
    double migrated;
    double mem_press;
    double numad_cpu;
};

// Output metrics - numad tuning parameters, relevant for the task
struct NumadParams {
    double u; // Target utilization % (50 - 130)
    double m; // Target memory locality % (50 - 100)
    double H; // THP scan sleep ms (0 - 5000)
};

class NumadBayesianOptimizer {
private:
    std::vector<NumadParams> X_sample;
    std::vector<double> Y_sample;
    
    // Hyperparameters for Gaussian Process
    const double l = 25.0;       // Length-scale
    const double sigma_f = 10.0;  // Signal variance
    const double sigma_n = 0.1;   // Noise variance
    const double kappa = 2.0;     // UCB exploration parameter
    const size_t max_history = 100; // Sliding window history

    // Score function normalization weights
    const double alpha = 5.0;  // MISS penalty
    const double beta = 2.0;   // OTHER penalty
    const double gamma = 1.5;  // MIGRATION stability penalty
    const double delta = 10.0; // MEM_PRESS critical penalty
    const double epsilon = 0.5; // CPU overhead penalty

    double compute_score(const SystemMetrics& m) {
        return m.numad_hit - (alpha * m.numa_miss + beta * m.numa_other + 
                              gamma * m.migrated + delta * m.mem_press + epsilon * m.numad_cpu);
    }

    double kernel(const NumadParams& a, const NumadParams& b) {
        double du = (a.u - b.u);
        double dm = (a.m - b.m);
        double dH = (a.H - b.H) / 50.0; // Scale down H to match dimensions
        double dist_sq = du*du + dm*dm + dH*dH;
        return sigma_f * sigma_f * std::exp(-dist_sq / (2.0 * l * l));
    }

    std::vector<std::vector<double>> invert_matrix(const std::vector<std::vector<double>>& A) {
        size_t n = A.size();
        std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> temp = A;
        
        for (size_t i = 0; i < n; ++i) inverse[i][i] = 1.0;

        for (size_t i = 0; i < n; ++i) {
            double pivot = temp[i][i];
            for (size_t j = 0; j < n; ++j) {
                temp[i][j] /= pivot;
                inverse[i][j] /= pivot;
            }
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = temp[k][i];
                    for (size_t j = 0; j < n; ++j) {
                        temp[k][j] -= factor * temp[i][j];
                        inverse[k][j] -= factor * inverse[i][j];
                    }
                }
            }
        }
        return inverse;
    }

public:
    NumadBayesianOptimizer() {}

    void train(const NumadParams& params, const SystemMetrics& metrics) {
        double score = compute_score(metrics);
        
        if (X_sample.size() >= max_history) {
            X_sample.erase(X_sample.begin());
            Y_sample.erase(Y_sample.begin());
        }
        X_sample.push_back(params);
        Y_sample.push_back(score);
    }

    NumadParams run() {
        if (X_sample.empty()) {
            return NumadParams{85.0, 90.0, 1000.0};
        }

        size_t n = X_sample.size();
        std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                K[i][j] = kernel(X_sample[i], X_sample[j]);
                if (i == j) K[i][j] += sigma_n * sigma_n;
            }
        }

        std::vector<std::vector<double>> K_inv = invert_matrix(K);

        const int search_steps = 10000;
        NumadParams best_params{85.0, 90.0, 1000.0};
        double max_ucb = -std::numeric_limits<double>::infinity();

        #pragma omp parallel
        {
            std::mt19937 generator(omp_get_thread_num() + std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_real_distribution<double> dist_u(50.0, 130.0);
            std::uniform_real_distribution<double> dist_m(50.0, 100.0);
            std::uniform_real_distribution<double> dist_H(0.0, 5000.0);

            NumadParams thread_best_params{85.0, 90.0, 1000.0};
            double thread_max_ucb = -std::numeric_limits<double>::infinity();

            #pragma omp for nowait
            for (int i = 0; i < search_steps; ++i) {
                NumadParams candidate{dist_u(generator), dist_m(generator), dist_H(generator)};

                std::vector<double> k_star(n, 0.0);
                for (size_t j = 0; j < n; ++j) {
                    k_star[j] = kernel(X_sample[j], candidate);
                }

                double mu = 0.0;
                for (size_t r = 0; r < n; ++r) {
                    double inv_sum = 0.0;
                    for (size_t c = 0; c < n; ++c) {
                        inv_sum += K_inv[r][c] * Y_sample[c];
                    }
                    mu += k_star[r] * inv_sum;
                }

                double k_ss = kernel(candidate, candidate);
                double variance_reduction = 0.0;
                for (size_t r = 0; r < n; ++r) {
                    double inv_sum = 0.0;
                    for (size_t c = 0; c < n; ++c) {
                        inv_sum += K_inv[r][c] * k_star[c];
                    }
                    variance_reduction += k_star[r] * inv_sum;
                }
                double sigma = std::sqrt(std::max(0.0, k_ss - variance_reduction));

                double ucb = mu + kappa * sigma;

                if (ucb > thread_max_ucb) {
                    thread_max_ucb = ucb;
                    thread_best_params = candidate;
                }
            }

            #pragma omp critical
            {
                if (thread_max_ucb > max_ucb) {
                    max_ucb = thread_max_ucb;
                    best_params = thread_best_params;
                }
            }
        }

        return best_params;
    }
};

// Simulator class modeling asymmetric NUMA node pressure reactions
class HardwareNUMASimulator {
private:
    std::mt19937 rng;
    double time_step = 0.0;

public:
    HardwareNUMASimulator() : rng(42) {}

    SystemMetrics generate_metrics(const NumadParams& p, bool optimized_mode) {
        time_step += 0.1;
        std::normal_distribution<double> noise(0.0, 2.0);

        SystemMetrics m;
        if (!optimized_mode) {
            m.numad_hit = 60.0 + noise(rng);
            m.numa_miss = 15.0 + std::abs(noise(rng));
            m.numa_other = 20.0 + std::abs(noise(rng));
            m.migrated = 30.0 + 5.0 * std::sin(time_step);
            m.mem_press = 12.0 + std::abs(noise(rng));
            m.numad_cpu = 5.0;
        } else {
            double penalty_m = (p.m > 95.0) ? (p.m - 95.0) * 4.0 : 0.0;
            double penalty_u = (p.u > 100.0) ? (p.u - 100.0) * 1.5 : (85.0 - p.u) * 0.5;
            double thp_efficiency = (p.H < 500.0 && p.H > 0) ? 15.0 : 0.0;
            double thp_cpu_drain = (p.H < 200.0 && p.H > 0) ? (200.0 - p.H) * 0.05 : 0.0;

            m.numad_hit = std::min(100.0, 85.0 + thp_efficiency - penalty_u + noise(rng));
            m.numa_miss = std::max(0.0, 8.0 - (p.m * 0.08) + penalty_m * 0.2 + noise(rng)*0.2);
            m.numa_other = std::max(0.0, 10.0 - (p.u * 0.05) + noise(rng)*0.2);
            m.migrated = std::max(0.0, 5.0 + penalty_m + noise(rng)*0.5);
            m.mem_press = std::max(0.0, penalty_u * 0.5 + noise(rng)*0.1);
            m.numad_cpu = std::max(0.5, 2.0 + thp_cpu_drain + (p.m * 0.02));
        }
        return m;
    }
};

int main() {
    NumadBayesianOptimizer optimizer;
    HardwareNUMASimulator cluster;
    
    std::ofstream csv_file("/Users/egor/Downloads/HPC_numad/numad_benchmark_results.csv");
    csv_file << "Cycle,Mode,HitRate,Misses,Other,Migrated,MemPress,CpuUsage,Param_u,Param_m,Param_H\n";

    NumadParams current_tuning{85.0, 90.0, 1000.0}; 
    const int evaluation_cycles = 60;

    std::cout << "Starting" << std::endl;

    for (int cycle = 1; cycle <= evaluation_cycles; ++cycle) {
        NumadParams static_config{85.0, 90.0, 1000.0};
        SystemMetrics unopt_metrics = cluster.generate_metrics(static_config, false);
        csv_file << cycle << ",Static_NoNumad," << unopt_metrics.numad_hit << "," << unopt_metrics.numa_miss << ","
                 << unopt_metrics.numa_other << "," << unopt_metrics.migrated << "," << unopt_metrics.mem_press << ","
                 << unopt_metrics.numad_cpu << "," << static_config.u << "," << static_config.m << "," << static_config.H << "\n";

        SystemMetrics opt_metrics = cluster.generate_metrics(current_tuning, true);
        
        optimizer.train(current_tuning, opt_metrics);
        current_tuning = optimizer.run();

        csv_file << cycle << ",Bayesian_Optimized," << opt_metrics.numad_hit << "," << opt_metrics.numa_miss << ","
                 << opt_metrics.numa_other << "," << opt_metrics.migrated << "," << opt_metrics.mem_press << ","
                 << opt_metrics.numad_cpu << "," << current_tuning.u << "," << current_tuning.m << "," << current_tuning.H << "\n";
    }

    csv_file.close();
    std::cout << "Done. Proceed with the plotter file" << std::endl;
    return 0;
}