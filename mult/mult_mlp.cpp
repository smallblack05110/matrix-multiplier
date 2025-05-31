#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>

// 编译：hipcc mult_mlp.cpp -o mlp_full_dcu
// 运行：./mlp_full_dcu

// ============================ 配置参数 ============================

#define INPUT_DIM     10
#define HIDDEN_DIM1   256
#define HIDDEN_DIM2   128
#define HIDDEN_DIM3   64
#define OUTPUT_DIM    1
#define BATCH_SIZE    64
#define EPOCHS        1200
#define LEARNING_RATE 1e-3
#define WEIGHT_DECAY  5e-5
#define DROPOUT_RATE  0.2

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================ GPU Kernels ============================

// 矩阵乘法 (M×K) × (K×N) = (M×N)，
// 参数顺序：A(M×K)，B(K×N)，C(M×N)，然后 M, N, K。
__global__ void matmul_optimized(const double* A, const double* B, double* C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Leaky ReLU 前向
__global__ void leaky_relu_forward(const double* input, double* output, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        output[idx] = (x > 0.0) ? x : alpha * x;
    }
}

// Leaky ReLU 反向
__global__ void leaky_relu_backward(double* grad, const double* input, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        grad[idx] *= (x > 0.0) ? 1.0 : alpha;
    }
}

// 梯度裁剪：将 grad 中的值裁剪到 [-max_norm, +max_norm]
__global__ void clip_gradients(double* grad, int size, double max_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (grad[idx] > max_norm) grad[idx] = max_norm;
        else if (grad[idx] < -max_norm) grad[idx] = -max_norm;
    }
}

// 加偏置：output[b*cols + col] += bias[col]
__global__ void add_bias(double* output, const double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

// 计算输出层梯度：grad[i] = 2*(pred[i] - target[i]) / size
__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / size;
    }
}

// AdamW 更新权重矩阵（带动量、带权重衰减）
__global__ void adamw_update(double* weights, const double* grad, double* m, double* v,
                             double lr, double beta1, double beta2, double eps, double weight_decay,
                             int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
        double m_hat = m[idx] / (1.0 - pow(beta1, t));
        double v_hat = v[idx] / (1.0 - pow(beta2, t));
        weights[idx] = weights[idx] * (1.0 - lr * weight_decay)
                     - lr * m_hat / (sqrt(v_hat) + eps);
    }
}

// 矩阵转置：将 input(rows×cols) 转成 output(cols×rows)
__global__ void transpose_matrix(const double* input, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int r = idx / cols;
        int c = idx % cols;
        output[c * rows + r] = input[r * cols + c];
    }
}

// 计算每个特征维度的偏置梯度：grad_bias[i] = sum_{b=0..batch-1}(grad_output[b*features + i])
__global__ void compute_bias_grad(const double* grad_output, double* grad_bias, int batch_size, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < features) {
        double sum = 0.0;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * features + idx];
        }
        grad_bias[idx] = sum;
    }
}

// Cosine Annealing with Warm Restarts 学习率调度（Host/Device 共用）
__host__ __device__
double cosine_annealing_warm_restart(double initial_lr, int epoch, int max_epochs) {
    double T_cur = epoch % (max_epochs / 4);
    double T_i   = max_epochs / 4;
    return initial_lr * 0.5 * (1.0 + cos(M_PI * T_cur / T_i));
}

// 对偏置用简单 SGD 更新：bias[i] -= lr * grad_bias[i]
__global__ void bias_sgd_update(double* bias, const double* grad_bias, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        bias[idx] -= lr * grad_bias[idx];
    }
}

// ============================ Host 辅助函数 ============================

// 生成3000个带宽数据（多频率叠加 + 非线性趋势 + 噪声）
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(42);
    std::normal_distribution<> noise(0.0, 20.0);

    for (int i = 0; i < 3000; i++) {
        double t = i * 0.01;
        double base = 500.0;
        double trend = 0.03 * i + 0.0001 * i * i;
        double seasonal1 = 180.0 * sin(t * 0.2);
        double seasonal2 = 120.0 * sin(t * 0.8);
        double seasonal3 = 60.0 * sin(t * 2.5);
        double seasonal4 = 30.0 * sin(t * 8.0);
        double seasonal5 = 15.0 * sin(t * 20.0);
        double random_noise = noise(gen);
        if (i % 400 == 0) {
            random_noise += 100.0 * ((gen() % 2 == 0) ? 1.0 : -1.0);
        }
        double value = base + trend + seasonal1 + seasonal2 + seasonal3 + seasonal4 + seasonal5 + random_noise;
        data.push_back(std::max(100.0, value));
    }
    std::cout << "[INFO] Generated " << data.size() << " complex bandwidth data points" << std::endl;
    return data;
}

// 创建滑动窗口数据集：X (num_samples × INPUT_DIM)，y (num_samples)
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y) {
    int num_samples = data.size() - INPUT_DIM;
    X.resize(num_samples * INPUT_DIM);
    y.resize(num_samples);
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            X[i * INPUT_DIM + j] = data[i + j];
        }
        y[i] = data[i + INPUT_DIM];
    }
    std::cout << "[INFO] Created dataset with " << num_samples << " samples" << std::endl;
}

// 分位数归一化（Robust scaling）
void normalize_data_robust(std::vector<double>& data, double& q25, double& q75) {
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    int n = sorted_data.size();
    q25 = sorted_data[n / 4];
    q75 = sorted_data[3 * n / 4];
    double iqr = q75 - q25;
    for (auto& val : data) {
        val = (val - q25) / iqr;
    }
    std::cout << "[INFO] Data normalized (Robust): Q25=" << q25 << ", Q75=" << q75 << std::endl;
}

// 分位数反归一化
void denormalize_data_robust(std::vector<double>& data, double q25, double q75) {
    double iqr = q75 - q25;
    for (auto& val : data) {
        val = val * iqr + q25;
    }
}

// He 初始化
void initialize_weights_he(std::vector<double>& weights, int fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev = sqrt(2.0 / fan_in);
    std::normal_distribution<> dis(0.0, std_dev);
    for (int i = 0; i < (int)weights.size(); i++) {
        weights[i] = dis(gen);
    }
}

// 计算回归指标：MSE, MAE, R², MAPE
void compute_metrics(const std::vector<double>& pred, const std::vector<double>& actual,
                     double& mse, double& mae, double& r2, double& mape) {
    mse = mae = mape = 0.0;
    double ss_res = 0.0, ss_tot = 0.0;
    double mean_actual = 0.0;
    for (double v : actual) mean_actual += v;
    mean_actual /= actual.size();
    for (size_t i = 0; i < pred.size(); i++) {
        double diff = pred[i] - actual[i];
        mse  += diff * diff;
        mae  += fabs(diff);
        mape += fabs(diff / actual[i]) * 100.0;
        ss_res += diff * diff;
        ss_tot += (actual[i] - mean_actual) * (actual[i] - mean_actual);
    }
    mse /= pred.size();
    mae /= pred.size();
    mape /= pred.size();
    r2 = 1.0 - (ss_res / ss_tot);
}

// 保存四层网络模型到文本文件
void save_model_4layer(const std::vector<double>& w1, const std::vector<double>& b1,
                       const std::vector<double>& w2, const std::vector<double>& b2,
                       const std::vector<double>& w3, const std::vector<double>& b3,
                       const std::vector<double>& w4, const std::vector<double>& b4,
                       double norm_param1, double norm_param2, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[WARN] Cannot open file to save model: " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(10);
    file << norm_param1 << " " << norm_param2 << "\n";
    for (double v : w1) file << v << " ";
    file << "\n";
    for (double v : b1) file << v << " ";
    file << "\n";
    for (double v : w2) file << v << " ";
    file << "\n";
    for (double v : b2) file << v << " ";
    file << "\n";
    for (double v : w3) file << v << " ";
    file << "\n";
    for (double v : b3) file << v << " ";
    file << "\n";
    for (double v : w4) file << v << " ";
    file << "\n";
    for (double v : b4) file << v << " ";
    file << "\n";
    file.close();
    std::cout << "[INFO] 4-layer model saved to " << filename << std::endl;
}

// ============================ Main Function ============================

int main() {
    // 强制使用 GPU 设备 0
    hipSetDevice(0);

    // 读取并生成带宽数据
    std::vector<double> bandwidth_data = load_json_bandwidth("bandwidth.json");

    // 分位数归一化
    double q25, q75;
    normalize_data_robust(bandwidth_data, q25, q75);

    // 创建滑动窗口数据集
    std::vector<double> X, y;
    create_dataset(bandwidth_data, X, y);
    int num_samples = (int)y.size();
    int train_size  = static_cast<int>(num_samples * 0.9);

    // 划分训练/测试集
    std::vector<double> X_train(X.begin(), X.begin() + train_size * INPUT_DIM);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<double> X_test(X.begin() + train_size * INPUT_DIM, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());

    std::cout << "[INFO] Training samples: " << train_size
              << ", Test samples: " << y_test.size() << std::endl;

    // 在 Host 端分配并初始化网络参数（4 层）
    std::vector<double> h_w1(INPUT_DIM * HIDDEN_DIM1),  h_b1(HIDDEN_DIM1, 0.0);
    std::vector<double> h_w2(HIDDEN_DIM1 * HIDDEN_DIM2), h_b2(HIDDEN_DIM2, 0.0);
    std::vector<double> h_w3(HIDDEN_DIM2 * HIDDEN_DIM3), h_b3(HIDDEN_DIM3, 0.0);
    std::vector<double> h_w4(HIDDEN_DIM3 * OUTPUT_DIM),  h_b4(OUTPUT_DIM,  0.0);

    initialize_weights_he(h_w1, INPUT_DIM);
    initialize_weights_he(h_w2, HIDDEN_DIM1);
    initialize_weights_he(h_w3, HIDDEN_DIM2);
    initialize_weights_he(h_w4, HIDDEN_DIM3);

    // 为权重矩阵分配 AdamW 的动量参数
    std::vector<double> m_w1(INPUT_DIM * HIDDEN_DIM1, 0.0), v_w1(INPUT_DIM * HIDDEN_DIM1, 0.0);
    std::vector<double> m_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0), v_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0);
    std::vector<double> m_w3(HIDDEN_DIM2 * HIDDEN_DIM3, 0.0), v_w3(HIDDEN_DIM2 * HIDDEN_DIM3, 0.0);
    std::vector<double> m_w4(HIDDEN_DIM3 * OUTPUT_DIM,  0.0), v_w4(HIDDEN_DIM3 * OUTPUT_DIM,  0.0);

    // 在 GPU 上分配所有必要的内存
    double *d_X, *d_y;
    double *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_w4, *d_b4;
    double *d_h1, *d_a1, *d_h2, *d_a2, *d_h3, *d_a3, *d_output;
    double *d_grad_output, *d_grad_h3, *d_grad_h2, *d_grad_h1;
    double *d_grad_w1, *d_grad_b1, *d_grad_w2, *d_grad_b2, *d_grad_w3, *d_grad_b3, *d_grad_w4, *d_grad_b4;
    double *d_m_w1, *d_v_w1, *d_m_w2, *d_v_w2, *d_m_w3, *d_v_w3, *d_m_w4, *d_v_w4;
    double *d_temp1, *d_temp2; // 临时矩阵用于转置或中间存储

    int batch_samples = std::min(BATCH_SIZE, train_size);

    // 输入和目标
    hipMalloc(&d_X, batch_samples * INPUT_DIM * sizeof(double));
    hipMalloc(&d_y, batch_samples * sizeof(double));

    // 四层网络的权重和偏置
    hipMalloc(&d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_b1, HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_b2, HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_b3, HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_b4, OUTPUT_DIM * sizeof(double));

    // 隐藏层输出和激活，以及最终输出
    hipMalloc(&d_h1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_a1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_h2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_a2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_h3, batch_samples * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_a3, batch_samples * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_output, batch_samples * OUTPUT_DIM * sizeof(double));

    // 反向传播时各种梯度
    hipMalloc(&d_grad_output, batch_samples * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_grad_h3, batch_samples * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_grad_h2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_h1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_b1, HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_b2, HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_grad_b3, HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_grad_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_grad_b4, OUTPUT_DIM * sizeof(double));

    // 权重矩阵 AdamW 动量参数
    hipMalloc(&d_m_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_v_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_m_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_v_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_m_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_v_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_m_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_v_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));

    // 临时矩阵大小取所有层维度的最大值
    int max_dim = std::max({INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, HIDDEN_DIM3, OUTPUT_DIM});
    hipMalloc(&d_temp1, batch_samples * max_dim * sizeof(double));
    hipMalloc(&d_temp2, max_dim * max_dim * sizeof(double));

    // 把 Host 上初始化好的权重参数拷贝到 Device
    hipMemcpy(d_w1, h_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1.data(), HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2.data(), HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3.data(), HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w4, h_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b4, h_b4.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);

    // 把动量参数拷到 Device
    hipMemcpy(d_m_w1, m_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w1, v_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w2, m_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w2, v_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w3, m_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w3, v_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w4, m_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w4, v_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);

    std::cout << "[INFO] Starting training with improved 4-layer network..." << std::endl;

    // AdamW 的超参数
    double beta1    = 0.9, beta2 = 0.999, eps = 1e-8;
    double best_loss = 1e6;
    int patience    = 0;
    const int max_patience = 150;

    // 为了在 Host 端计算 MSE，我们准备一个 Host 缓冲区来拷贝 d_output
    double* h_pred_batch = new double[BATCH_SIZE];
    double* h_true_batch = new double[BATCH_SIZE];

    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;
        int num_batches = (train_size + batch_samples - 1) / batch_samples;

        // 学习率调度
        double current_lr = cosine_annealing_warm_restart(LEARNING_RATE, epoch, EPOCHS);

        for (int batch_start = 0; batch_start < train_size; batch_start += batch_samples) {
            int current_batch_size = std::min(batch_samples, train_size - batch_start);

            // 拷贝这一批的 X 和 y 到 GPU
            hipMemcpy(d_X, X_train.data() + batch_start * INPUT_DIM,
                     current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y, y_train.data() + batch_start,
                     current_batch_size * sizeof(double), hipMemcpyHostToDevice);

            // ================= 前向传播 =================

            dim3 block(16, 16);
            dim3 grid_layer1((HIDDEN_DIM1 + block.x - 1) / block.x,
                             (current_batch_size + block.y - 1) / block.y);
            dim3 grid_layer2((HIDDEN_DIM2 + block.x - 1) / block.x,
                             (current_batch_size + block.y - 1) / block.y);
            dim3 grid_layer3((HIDDEN_DIM3 + block.x - 1) / block.x,
                             (current_batch_size + block.y - 1) / block.y);
            dim3 grid_output((OUTPUT_DIM + block.x - 1) / block.x,
                             (current_batch_size + block.y - 1) / block.y);

            // Layer1: h1 = X @ W1  => (current_batch_size×INPUT_DIM) × (INPUT_DIM×HIDDEN_DIM1) = (current_batch_size×HIDDEN_DIM1)
            matmul_optimized<<<grid_layer1, block>>>(
                d_X, d_w1, d_h1,
                current_batch_size, HIDDEN_DIM1, INPUT_DIM
            );
            // + b1
            add_bias<<< (current_batch_size * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_h1, d_b1, current_batch_size, HIDDEN_DIM1
            );
            // LeakyReLU
            leaky_relu_forward<<< (current_batch_size * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_h1, d_a1, current_batch_size * HIDDEN_DIM1, 0.02
            );

            // Layer2: h2 = a1 @ W2 => (current_batch_size×HIDDEN_DIM1) × (HIDDEN_DIM1×HIDDEN_DIM2) = (current_batch_size×HIDDEN_DIM2)
            matmul_optimized<<<grid_layer2, block>>>(
                d_a1, d_w2, d_h2,
                current_batch_size, HIDDEN_DIM2, HIDDEN_DIM1
            );
            add_bias<<< (current_batch_size * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_h2, d_b2, current_batch_size, HIDDEN_DIM2
            );
            leaky_relu_forward<<< (current_batch_size * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_h2, d_a2, current_batch_size * HIDDEN_DIM2, 0.02
            );

            // Layer3: h3 = a2 @ W3 => (current_batch_size×HIDDEN_DIM2) × (HIDDEN_DIM2×HIDDEN_DIM3) = (current_batch_size×HIDDEN_DIM3)
            matmul_optimized<<<grid_layer3, block>>>(
                d_a2, d_w3, d_h3,
                current_batch_size, HIDDEN_DIM3, HIDDEN_DIM2
            );
            add_bias<<< (current_batch_size * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_h3, d_b3, current_batch_size, HIDDEN_DIM3
            );
            leaky_relu_forward<<< (current_batch_size * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_h3, d_a3, current_batch_size * HIDDEN_DIM3, 0.02
            );

            // Output: out = a3 @ W4 + b4 => (current_batch_size×HIDDEN_DIM3) × (HIDDEN_DIM3×OUTPUT_DIM) = (current_batch_size×OUTPUT_DIM)
            matmul_optimized<<<grid_output, block>>>(
                d_a3, d_w4, d_output,
                current_batch_size, OUTPUT_DIM, HIDDEN_DIM3
            );
            add_bias<<< (current_batch_size * OUTPUT_DIM + 255)/256, 256 >>>(
                d_output, d_b4, current_batch_size, OUTPUT_DIM
            );

            // ============== 在 Host 端计算 MSE ==============
            hipMemcpy(h_pred_batch, d_output, current_batch_size * sizeof(double), hipMemcpyDeviceToHost);
            for (int i = 0; i < current_batch_size; i++) {
                h_true_batch[i] = y_train[batch_start + i];
            }
            double batch_loss = 0.0;
            for (int i = 0; i < current_batch_size; i++) {
                double diff = h_pred_batch[i] - h_true_batch[i];
                batch_loss += diff * diff;
            }
            batch_loss /= current_batch_size;
            total_loss += batch_loss;

            // ============== 反向传播 ==============

            // 1. 输出层梯度 dL/dout
            compute_output_grad<<< (current_batch_size + 255)/256, 256 >>>(
                d_output, d_y, d_grad_output, current_batch_size
            );

            // 2. 第4 层（输出层）梯度计算
            dim3 block_bw(16, 16);

            // 2.1 转置 a3 --> temp1  (a3 为 current_batch_size×HIDDEN_DIM3) 转成 (HIDDEN_DIM3×current_batch_size)
            transpose_matrix<<< (current_batch_size * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_a3, d_temp1, current_batch_size, HIDDEN_DIM3
            );
            // dW4 = temp1(HIDDEN_DIM3×current_batch_size) @ d_grad_output(current_batch_size×OUTPUT_DIM)
            //      => (HIDDEN_DIM3×OUTPUT_DIM)
            int gx4 = (HIDDEN_DIM3 + block_bw.x - 1) / block_bw.x;
            int gy4 = (OUTPUT_DIM  + block_bw.y - 1) / block_bw.y;
            dim3 grid4(gx4, gy4);
            matmul_optimized<<< grid4, block_bw >>>(
                d_temp1, d_grad_output, d_grad_w4,
                HIDDEN_DIM3, OUTPUT_DIM, current_batch_size
            );

            // 2.2 db4 = sum(d_grad_output, axis=0)
            compute_bias_grad<<< (OUTPUT_DIM + 255)/256, 256 >>>(
                d_grad_output, d_grad_b4, current_batch_size, OUTPUT_DIM
            );

            // 3. grad_h3 = d_grad_output @ W4^T
            //    d_grad_output: (current_batch_size×OUTPUT_DIM)，W4^T: (OUTPUT_DIM×HIDDEN_DIM3)
            transpose_matrix<<< (HIDDEN_DIM3 * OUTPUT_DIM + 255)/256, 256 >>>(
                d_w4, d_temp2, HIDDEN_DIM3, OUTPUT_DIM
            );
            int gx_h3 = (current_batch_size + block_bw.x - 1) / block_bw.x;
            int gy_h3 = (HIDDEN_DIM3        + block_bw.y - 1) / block_bw.y;
            dim3 grid_h3(gx_h3, gy_h3);
            matmul_optimized<<< grid_h3, block_bw >>>(
                d_grad_output, d_temp2, d_grad_h3,
                current_batch_size, HIDDEN_DIM3, OUTPUT_DIM
            );

            // 4. LeakyReLU 反向 (第3层)
            leaky_relu_backward<<< (current_batch_size * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_grad_h3, d_h3, current_batch_size * HIDDEN_DIM3, 0.02
            );

            // 5. dW3 = a2^T @ d_grad_h3
            //    a2^T: (HIDDEN_DIM2×current_batch_size), d_grad_h3: (current_batch_size×HIDDEN_DIM3)
            transpose_matrix<<< (current_batch_size * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_a2, d_temp1, current_batch_size, HIDDEN_DIM2
            );
            int gx_w3 = (HIDDEN_DIM2 + block_bw.x - 1) / block_bw.x;
            int gy_w3 = (HIDDEN_DIM3 + block_bw.y - 1) / block_bw.y;
            dim3 grid_w3(gx_w3, gy_w3);
            matmul_optimized<<< grid_w3, block_bw >>>(
                d_temp1, d_grad_h3, d_grad_w3,
                HIDDEN_DIM2, HIDDEN_DIM3, current_batch_size
            );

            // 5.1 db3 = sum(d_grad_h3, axis=0)
            compute_bias_grad<<< (HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_grad_h3, d_grad_b3, current_batch_size, HIDDEN_DIM3
            );

            // 6. grad_h2 = d_grad_h3 @ W3^T
            //    d_grad_h3: (current_batch_size×HIDDEN_DIM3), W3^T: (HIDDEN_DIM3×HIDDEN_DIM2)
            transpose_matrix<<< (HIDDEN_DIM2 * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_w3, d_temp2, HIDDEN_DIM2, HIDDEN_DIM3
            );
            int gx_h2 = (current_batch_size + block_bw.x - 1) / block_bw.x;
            int gy_h2 = (HIDDEN_DIM2        + block_bw.y - 1) / block_bw.y;
            dim3 grid_h2(gx_h2, gy_h2);
            matmul_optimized<<< grid_h2, block_bw >>>(
                d_grad_h3, d_temp2, d_grad_h2,
                current_batch_size, HIDDEN_DIM2, HIDDEN_DIM3
            );

            // 7. LeakyReLU 反向 (第2层)
            leaky_relu_backward<<< (current_batch_size * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_grad_h2, d_h2, current_batch_size * HIDDEN_DIM2, 0.02
            );

            // 8. dW2 = a1^T @ d_grad_h2
            //    a1^T: (HIDDEN_DIM1×current_batch_size), d_grad_h2: (current_batch_size×HIDDEN_DIM2)
            transpose_matrix<<< (current_batch_size * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_a1, d_temp1, current_batch_size, HIDDEN_DIM1
            );
            int gx_w2 = (HIDDEN_DIM1 + block_bw.x - 1) / block_bw.x;
            int gy_w2 = (HIDDEN_DIM2 + block_bw.y - 1) / block_bw.y;
            dim3 grid_w2(gx_w2, gy_w2);
            matmul_optimized<<< grid_w2, block_bw >>>(
                d_temp1, d_grad_h2, d_grad_w2,
                HIDDEN_DIM1, HIDDEN_DIM2, current_batch_size
            );

            // 8.1 db2 = sum(d_grad_h2, axis=0)
            compute_bias_grad<<< (HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_grad_h2, d_grad_b2, current_batch_size, HIDDEN_DIM2
            );

            // 9. grad_h1 = d_grad_h2 @ W2^T
            //    d_grad_h2: (current_batch_size×HIDDEN_DIM2), W2^T: (HIDDEN_DIM2×HIDDEN_DIM1)
            transpose_matrix<<< (HIDDEN_DIM1 * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_w2, d_temp2, HIDDEN_DIM1, HIDDEN_DIM2
            );
            int gx_h1 = (current_batch_size + block_bw.x - 1) / block_bw.x;
            int gy_h1 = (HIDDEN_DIM1        + block_bw.y - 1) / block_bw.y;
            dim3 grid_h1(gx_h1, gy_h1);
            matmul_optimized<<< grid_h1, block_bw >>>(
                d_grad_h2, d_temp2, d_grad_h1,
                current_batch_size, HIDDEN_DIM1, HIDDEN_DIM2
            );

            // 10. LeakyReLU 反向 (第1层)
            leaky_relu_backward<<< (current_batch_size * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_grad_h1, d_h1, current_batch_size * HIDDEN_DIM1, 0.02
            );

            // 11. dW1 = X^T @ d_grad_h1
            //     X^T: (INPUT_DIM×current_batch_size), d_grad_h1: (current_batch_size×HIDDEN_DIM1)
            transpose_matrix<<< (current_batch_size * INPUT_DIM + 255)/256, 256 >>>(
                d_X, d_temp1, current_batch_size, INPUT_DIM
            );
            int gx_w1 = (INPUT_DIM    + block_bw.x - 1) / block_bw.x;
            int gy_w1 = (HIDDEN_DIM1  + block_bw.y - 1) / block_bw.y;
            dim3 grid_w1(gx_w1, gy_w1);
            matmul_optimized<<< grid_w1, block_bw >>>(
                d_temp1, d_grad_h1, d_grad_w1,
                INPUT_DIM, HIDDEN_DIM1, current_batch_size
            );

            // 11.1 db1 = sum(d_grad_h1, axis=0)
            compute_bias_grad<<< (HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_grad_h1, d_grad_b1, current_batch_size, HIDDEN_DIM1
            );

            // ============== 梯度裁剪 ==============
            double grad_clip_norm = 5.0;
            clip_gradients<<< (INPUT_DIM * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_grad_w1,   INPUT_DIM * HIDDEN_DIM1,   grad_clip_norm
            );
            clip_gradients<<< (HIDDEN_DIM1 * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_grad_w2,   HIDDEN_DIM1 * HIDDEN_DIM2, grad_clip_norm
            );
            clip_gradients<<< (HIDDEN_DIM2 * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_grad_w3,   HIDDEN_DIM2 * HIDDEN_DIM3, grad_clip_norm
            );
            clip_gradients<<< (HIDDEN_DIM3 * OUTPUT_DIM + 255)/256, 256 >>>(
                d_grad_w4,   HIDDEN_DIM3 * OUTPUT_DIM,   grad_clip_norm
            );

            // ============== 优化器更新 ==============

            // W1 更新 (AdamW)
            adamw_update<<< (INPUT_DIM * HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_w1, d_grad_w1, d_m_w1, d_v_w1,
                current_lr, beta1, beta2, eps, WEIGHT_DECAY,
                epoch + 1, INPUT_DIM * HIDDEN_DIM1
            );
            // b1 更新 (SGD)
            bias_sgd_update<<< (HIDDEN_DIM1 + 255)/256, 256 >>>(
                d_b1, d_grad_b1, current_lr, HIDDEN_DIM1
            );

            // W2 更新 (AdamW)
            adamw_update<<< (HIDDEN_DIM1 * HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_w2, d_grad_w2, d_m_w2, d_v_w2,
                current_lr, beta1, beta2, eps, WEIGHT_DECAY,
                epoch + 1, HIDDEN_DIM1 * HIDDEN_DIM2
            );
            // b2 更新 (SGD)
            bias_sgd_update<<< (HIDDEN_DIM2 + 255)/256, 256 >>>(
                d_b2, d_grad_b2, current_lr, HIDDEN_DIM2
            );

            // W3 更新 (AdamW)
            adamw_update<<< (HIDDEN_DIM2 * HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_w3, d_grad_w3, d_m_w3, d_v_w3,
                current_lr, beta1, beta2, eps, WEIGHT_DECAY,
                epoch + 1, HIDDEN_DIM2 * HIDDEN_DIM3
            );
            // b3 更新 (SGD)
            bias_sgd_update<<< (HIDDEN_DIM3 + 255)/256, 256 >>>(
                d_b3, d_grad_b3, current_lr, HIDDEN_DIM3
            );

            // W4 更新 (AdamW)
            adamw_update<<< (HIDDEN_DIM3 * OUTPUT_DIM + 255)/256, 256 >>>(
                d_w4, d_grad_w4, d_m_w4, d_v_w4,
                current_lr, beta1, beta2, eps, WEIGHT_DECAY,
                epoch + 1, HIDDEN_DIM3 * OUTPUT_DIM
            );
            // b4 更新 (SGD)
            bias_sgd_update<<< (OUTPUT_DIM + 255)/256, 256 >>>(
                d_b4, d_grad_b4, current_lr, OUTPUT_DIM
            );
        } // end for batch

        total_loss /= num_batches;

        // 早停检测
        if (total_loss < best_loss - 1e-6) {
            best_loss = total_loss;
            patience   = 0;
        } else {
            patience++;
        }

        auto end_time  = std::chrono::high_resolution_clock::now();
        auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if ((epoch + 1) % 50 == 0) {
            std::cout << "[EPOCH " << std::setw(4) << epoch + 1 << "] Loss: "
                      << std::fixed << std::setprecision(6) << total_loss
                      << " | LR: " << std::scientific << current_lr
                      << " | Time: " << duration.count() << "ms"
                      << " | Patience: " << patience << std::endl;
        }

        if (patience >= max_patience) {
            std::cout << "[INFO] Early stopping triggered at epoch " << epoch + 1 << std::endl;
            break;
        }
        if ((epoch + 1) % 200 == 0) {
            beta1 = std::max(0.85, beta1 * 0.98);
            beta2 = std::max(0.99, beta2 * 0.999);
        }
    } // end for epoch

    std::cout << "[INFO] Training completed. Best loss: " << best_loss << std::endl;

    // ========================== 测试阶段 ==========================
    // 拷贝最终参数到 Host
    hipMemcpy(h_w1.data(), d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b1.data(), d_b1, HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w2.data(), d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b2.data(), d_b2, HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w3.data(), d_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b3.data(), d_b3, HIDDEN_DIM3 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w4.data(), d_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b4.data(), d_b4, OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);

    std::vector<double> test_predictions;
    test_predictions.reserve(y_test.size());

    std::cout << "[INFO] Testing model..." << std::endl;

    // Host 端逐样本推理
    for (size_t i = 0; i < y_test.size(); i++) {
        // 取一条样本
        std::vector<double> input(X_test.begin() + i * INPUT_DIM,
                                  X_test.begin() + (i + 1) * INPUT_DIM);

        // Layer1 (CPU)
        std::vector<double> h1(HIDDEN_DIM1, 0.0);
        for (int j = 0; j < HIDDEN_DIM1; j++) {
            for (int k = 0; k < INPUT_DIM; k++) {
                h1[j] += input[k] * h_w1[k * HIDDEN_DIM1 + j];
            }
            h1[j] += h_b1[j];
            h1[j] = (h1[j] > 0.0) ? h1[j] : 0.02 * h1[j];
        }
        // Layer2
        std::vector<double> h2(HIDDEN_DIM2, 0.0);
        for (int j = 0; j < HIDDEN_DIM2; j++) {
            for (int k = 0; k < HIDDEN_DIM1; k++) {
                h2[j] += h1[k] * h_w2[k * HIDDEN_DIM2 + j];
            }
            h2[j] += h_b2[j];
            h2[j] = (h2[j] > 0.0) ? h2[j] : 0.02 * h2[j];
        }
        // Layer3
        std::vector<double> h3(HIDDEN_DIM3, 0.0);
        for (int j = 0; j < HIDDEN_DIM3; j++) {
            for (int k = 0; k < HIDDEN_DIM2; k++) {
                h3[j] += h2[k] * h_w3[k * HIDDEN_DIM3 + j];
            }
            h3[j] += h_b3[j];
            h3[j] = (h3[j] > 0.0) ? h3[j] : 0.02 * h3[j];
        }
        // Output
        double out_val = 0.0;
        for (int k = 0; k < HIDDEN_DIM3; k++) {
            out_val += h3[k] * h_w4[k];
        }
        out_val += h_b4[0];

        test_predictions.push_back(out_val);
    }

    // 反归一化
    std::vector<double> denorm_predictions = test_predictions;
    std::vector<double> denorm_actual      = y_test;
    denormalize_data_robust(denorm_predictions, q25, q75);
    denormalize_data_robust(denorm_actual,      q25, q75);

    // 计算指标
    double mse, mae, r2, mape;
    compute_metrics(denorm_predictions, denorm_actual, mse, mae, r2, mape);

    std::cout << "\n=============== TEST RESULTS ===============" << std::endl;
    std::cout << "MSE:  " << std::fixed << std::setprecision(2) << mse << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(2) << sqrt(mse) << std::endl;
    std::cout << "MAE:  " << std::fixed << std::setprecision(2) << mae << std::endl;
    std::cout << "R²:   " << std::fixed << std::setprecision(4) << r2  << std::endl;
    std::cout << "MAPE: " << std::fixed << std::setprecision(2) << mape << "%" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << "\nSample predictions (first 10):" << std::endl;
    std::cout << "Predicted -> Actual" << std::endl;
    for (int i = 0; i < std::min(10, (int)denorm_predictions.size()); i++) {
        std::cout << std::fixed << std::setprecision(2)
                  << denorm_predictions[i] << " -> " << denorm_actual[i] << std::endl;
    }

    // 保存模型到文件
    save_model_4layer(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3, h_w4, h_b4,
                      q25, q75, "4layer_bandwidth_model.txt");

    // 释放 GPU 内存
    hipFree(d_X);      hipFree(d_y);
    hipFree(d_w1);     hipFree(d_b1);
    hipFree(d_w2);     hipFree(d_b2);
    hipFree(d_w3);     hipFree(d_b3);
    hipFree(d_w4);     hipFree(d_b4);
    hipFree(d_h1);     hipFree(d_a1);
    hipFree(d_h2);     hipFree(d_a2);
    hipFree(d_h3);     hipFree(d_a3);
    hipFree(d_output);
    hipFree(d_grad_output);
    hipFree(d_grad_h3); hipFree(d_grad_h2); hipFree(d_grad_h1);
    hipFree(d_grad_w1); hipFree(d_grad_b1);
    hipFree(d_grad_w2); hipFree(d_grad_b2);
    hipFree(d_grad_w3); hipFree(d_grad_b3);
    hipFree(d_grad_w4); hipFree(d_grad_b4);
    hipFree(d_m_w1);    hipFree(d_v_w1);
    hipFree(d_m_w2);    hipFree(d_v_w2);
    hipFree(d_m_w3);    hipFree(d_v_w3);
    hipFree(d_m_w4);    hipFree(d_v_w4);
    hipFree(d_temp1);   hipFree(d_temp2);

    delete[] h_pred_batch;
    delete[] h_true_batch;

    std::cout << "[INFO] Training and testing completed successfully!" << std::endl;
    return 0;
}
