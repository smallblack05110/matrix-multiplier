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

// 编译文件
// hipcc mult_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 优化后的参数配置
#define INPUT_DIM 10
#define HIDDEN_DIM1 128     // 增加第一个隐藏层
#define HIDDEN_DIM2 64      // 增加第二个隐藏层
#define HIDDEN_DIM3 32      // 添加第三个隐藏层
#define OUTPUT_DIM 1
#define BATCH_SIZE 32       // 进一步减小batch size
#define EPOCHS 800          // 增加训练轮数
#define LEARNING_RATE 5e-4  // 降低学习率避免震荡
#define WEIGHT_DECAY 1e-4   // 添加L2正则化

// 改进的HIP kernels
__global__ void matmul_optimized(const double* A, const double* B, double* C, int M, int N, int K) {
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

__global__ void swish_forward(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        output[idx] = x / (1.0 + exp(-x)); // Swish: x * sigmoid(x)
    }
}

__global__ void swish_backward(double* grad, const double* input, const double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double x = input[idx];
        double sigmoid_x = 1.0 / (1.0 + exp(-x));
        double swish_derivative = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x));
        grad[idx] = grad[idx] * swish_derivative;
    }
}

__global__ void add_bias(double* output, const double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]) / size;
    }
}

__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        atomicAdd(loss, diff * diff);
    }
}

// 改进的Adam优化器，加入权重衰减
__global__ void adam_update_with_decay(double* weights, const double* grad, double* m, double* v, 
                                      double lr, double beta1, double beta2, double eps, double weight_decay, 
                                      int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 添加L2正则化梯度
        double grad_with_decay = grad[idx] + weight_decay * weights[idx];
        
        // 更新动量
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad_with_decay;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad_with_decay * grad_with_decay;
        
        // 偏差修正
        double m_hat = m[idx] / (1.0 - pow(beta1, t));
        double v_hat = v[idx] / (1.0 - pow(beta2, t));
        
        // 更新权重
        weights[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

__global__ void transpose_matrix(const double* input, double* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[row * cols + col];
    }
}

// 改进的偏置梯度计算
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

// 学习率调度器
__device__ double learning_rate_schedule(double initial_lr, int epoch, int total_epochs) {
    // 余弦退火调度
    return initial_lr * 0.5 * (1.0 + cos(M_PI * epoch / total_epochs));
}

// 生成更复杂的模拟带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以获得可重复结果
    std::normal_distribution<> noise(0.0, 30.0); // 减少噪声
    
    // 生成更复杂的时间序列模式
    for (int i = 0; i < 2500; i++) {  // 增加数据量
        double t = i * 0.01;
        // 多频率叠加 + 趋势 + 噪声
        double base = 500.0;
        double trend = 0.05 * i; // 缓慢上升趋势
        double seasonal1 = 150.0 * sin(t * 0.3);      // 长周期
        double seasonal2 = 80.0 * sin(t * 1.5);       // 中周期  
        double seasonal3 = 40.0 * sin(t * 6.0);       // 短周期
        double seasonal4 = 20.0 * sin(t * 15.0);      // 高频周期
        double random_noise = noise(gen);
        
        // 添加一些突发事件（减少频率）
        if (i % 300 == 0) {
            random_noise += 150.0 * (gen() % 2 == 0 ? 1 : -1);
        }
        
        double value = base + trend + seasonal1 + seasonal2 + seasonal3 + seasonal4 + random_noise;
        data.push_back(std::max(100.0, value)); // 确保带宽不为负
    }
    
    std::cout << "[INFO] Generated " << data.size() << " complex bandwidth data points" << std::endl;
    return data;
}

// 创建滑动窗口数据集
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

// 改进的数据归一化（Z-score标准化）
void normalize_data_zscore(std::vector<double>& data, double& mean_val, double& std_val) {
    // 计算均值
    mean_val = 0.0;
    for (const auto& val : data) {
        mean_val += val;
    }
    mean_val /= data.size();
    
    // 计算标准差
    std_val = 0.0;
    for (const auto& val : data) {
        std_val += (val - mean_val) * (val - mean_val);
    }
    std_val = sqrt(std_val / (data.size() - 1)); // 使用样本标准差
    
    // 标准化
    for (auto& val : data) {
        val = (val - mean_val) / std_val;
    }
    
    std::cout << "[INFO] Data normalized (Z-score): mean=" << mean_val << ", std=" << std_val << std::endl;
}

// Z-score反归一化
void denormalize_data_zscore(std::vector<double>& data, double mean_val, double std_val) {
    for (auto& val : data) {
        val = val * std_val + mean_val;
    }
}

// Xavier/Glorot初始化
void initialize_weights_xavier(std::vector<double>& weights, int fan_in, int fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev = sqrt(2.0 / (fan_in + fan_out));
    std::normal_distribution<> dis(0.0, std_dev);
    
    for (int i = 0; i < weights.size(); i++) {
        weights[i] = dis(gen);
    }
}

// 计算评估指标
void compute_metrics(const std::vector<double>& pred, const std::vector<double>& actual,
                    double& mse, double& mae, double& r2) {
    mse = 0.0;
    mae = 0.0;
    double ss_res = 0.0, ss_tot = 0.0;
    
    // 计算actual的均值
    double mean_actual = 0.0;
    for (const auto& val : actual) {
        mean_actual += val;
    }
    mean_actual /= actual.size();
    
    for (size_t i = 0; i < pred.size(); i++) {
        double diff = pred[i] - actual[i];
        mse += diff * diff;
        mae += abs(diff);
        ss_res += diff * diff;
        ss_tot += (actual[i] - mean_actual) * (actual[i] - mean_actual);
    }
    
    mse /= pred.size();
    mae /= pred.size();
    r2 = 1.0 - (ss_res / ss_tot); // R²系数
}

// 保存模型（四层网络版本）
void save_model_4layer(const std::vector<double>& w1, const std::vector<double>& b1,
                       const std::vector<double>& w2, const std::vector<double>& b2,
                       const std::vector<double>& w3, const std::vector<double>& b3,
                       const std::vector<double>& w4, const std::vector<double>& b4,
                       double norm_param1, double norm_param2, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(10);
        file << norm_param1 << " " << norm_param2 << std::endl;
        
        // 保存所有层的权重和偏置
        for (const auto& w : w1) file << w << " ";
        file << std::endl;
        for (const auto& b : b1) file << b << " ";
        file << std::endl;
        for (const auto& w : w2) file << w << " ";
        file << std::endl;
        for (const auto& b : b2) file << b << " ";
        file << std::endl;
        for (const auto& w : w3) file << w << " ";
        file << std::endl;
        for (const auto& b : b3) file << b << " ";
        file << std::endl;
        for (const auto& w : w4) file << w << " ";
        file << std::endl;
        for (const auto& b : b4) file << b << " ";
        file << std::endl;
        
        file.close();
        std::cout << "[INFO] 4-layer model saved to " << filename << std::endl;
    }
}

// ----------------------------- Main -------------------------------
int main() {
    // 初始化HIP设备
    hipSetDevice(0);
    
    // 读取带宽数据
    std::vector<double> bandwidth_data = load_json_bandwidth("bandwidth.json");
    
    // 使用Z-score标准化
    double mean_val, std_val;
    normalize_data_zscore(bandwidth_data, mean_val, std_val);
    
    // 创建数据集
    std::vector<double> X, y;
    create_dataset(bandwidth_data, X, y);
    
    int num_samples = y.size();
    int train_size = static_cast<int>(num_samples * 0.85); // 增加训练集比例
    
    // 分割数据集
    std::vector<double> X_train(X.begin(), X.begin() + train_size * INPUT_DIM);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<double> X_test(X.begin() + train_size * INPUT_DIM, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());
    
    std::cout << "[INFO] Training samples: " << train_size << ", Test samples: " << y_test.size() << std::endl;
    
    // 初始化四层网络参数
    std::vector<double> h_w1(INPUT_DIM * HIDDEN_DIM1);
    std::vector<double> h_b1(HIDDEN_DIM1, 0.0);
    std::vector<double> h_w2(HIDDEN_DIM1 * HIDDEN_DIM2);
    std::vector<double> h_b2(HIDDEN_DIM2, 0.0);
    std::vector<double> h_w3(HIDDEN_DIM2 * HIDDEN_DIM3);
    std::vector<double> h_b3(HIDDEN_DIM3, 0.0);
    std::vector<double> h_w4(HIDDEN_DIM3 * OUTPUT_DIM);
    std::vector<double> h_b4(OUTPUT_DIM, 0.0);
    
    // 使用Xavier初始化
    initialize_weights_xavier(h_w1, INPUT_DIM, HIDDEN_DIM1);
    initialize_weights_xavier(h_w2, HIDDEN_DIM1, HIDDEN_DIM2);
    initialize_weights_xavier(h_w3, HIDDEN_DIM2, HIDDEN_DIM3);
    initialize_weights_xavier(h_w4, HIDDEN_DIM3, OUTPUT_DIM);
    
    // Adam优化器参数
    std::vector<double> m_w1(INPUT_DIM * HIDDEN_DIM1, 0.0);
    std::vector<double> v_w1(INPUT_DIM * HIDDEN_DIM1, 0.0);
    std::vector<double> m_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0);
    std::vector<double> v_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0);
    std::vector<double> m_w3(HIDDEN_DIM2 * HIDDEN_DIM3, 0.0);
    std::vector<double> v_w3(HIDDEN_DIM2 * HIDDEN_DIM3, 0.0);
    std::vector<double> m_w4(HIDDEN_DIM3 * OUTPUT_DIM, 0.0);
    std::vector<double> v_w4(HIDDEN_DIM3 * OUTPUT_DIM, 0.0);
    
    // GPU内存分配（扩展到四层网络）
    double *d_X, *d_y, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_w4, *d_b4;
    double *d_h1, *d_a1, *d_h2, *d_a2, *d_h3, *d_a3, *d_output, *d_loss;
    double *d_grad_output, *d_grad_h3, *d_grad_h2, *d_grad_h1;
    double *d_grad_w1, *d_grad_b1, *d_grad_w2, *d_grad_b2, *d_grad_w3, *d_grad_b3, *d_grad_w4, *d_grad_b4;
    double *d_m_w1, *d_v_w1, *d_m_w2, *d_v_w2, *d_m_w3, *d_v_w3, *d_m_w4, *d_v_w4;
    double *d_temp1, *d_temp2, *d_temp3; // 临时矩阵
    
    int batch_samples = std::min(BATCH_SIZE, train_size);
    
    // 分配内存
    hipMalloc(&d_X, batch_samples * INPUT_DIM * sizeof(double));
    hipMalloc(&d_y, batch_samples * sizeof(double));
    hipMalloc(&d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_b1, HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_b2, HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_b3, HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_b4, OUTPUT_DIM * sizeof(double));
    
    hipMalloc(&d_h1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_a1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_h2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_a2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_h3, batch_samples * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_a3, batch_samples * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_output, batch_samples * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_loss, sizeof(double));
    
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
    
    // Adam优化器内存
    hipMalloc(&d_m_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_v_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_m_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_v_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_m_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_v_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double));
    hipMalloc(&d_m_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_v_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double));
    
    // 临时矩阵
    int max_dim = std::max({INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, HIDDEN_DIM3, OUTPUT_DIM});
    hipMalloc(&d_temp1, batch_samples * max_dim * sizeof(double));
    hipMalloc(&d_temp2, max_dim * max_dim * sizeof(double));
    hipMalloc(&d_temp3, batch_samples * max_dim * sizeof(double));
    
    // 复制初始参数到GPU
    hipMemcpy(d_w1, h_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1.data(), HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2.data(), HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3.data(), HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w4, h_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b4, h_b4.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    
    // 复制Adam参数
    hipMemcpy(d_m_w1, m_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w1, v_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w2, m_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w2, v_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w3, m_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w3, v_w3.data(), HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w4, m_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w4, v_w4.data(), HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    
    std::cout << "[INFO] Starting training with improved 4-layer network..." << std::endl;
    
    // Adam优化器参数
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double best_loss = 1e6;
    int patience = 0;
    const int max_patience = 100;
    
    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;
        int num_batches = (train_size + batch_samples - 1) / batch_samples;
        
        // 动态学习率
        double current_lr = LEARNING_RATE * (1.0 - (double)epoch / EPOCHS) * 0.5 + LEARNING_RATE * 0.5;
        
        for (int batch_start = 0; batch_start < train_size; batch_start += batch_samples) {
            int current_batch_size = std::min(batch_samples, train_size - batch_start);
            
            // 复制批次数据
            hipMemcpy(d_X, X_train.data() + batch_start * INPUT_DIM, 
                     current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y, y_train.data() + batch_start, 
                     current_batch_size * sizeof(double), hipMemcpyHostToDevice);
            
            // 前向传播 - 四层网络
            dim3 block(16, 16);
            
            // 第一层: X * W1 + b1 -> Swish(h1)
            dim3 grid1((HIDDEN_DIM1 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid1, block>>>(d_X, d_w1, d_h1, current_batch_size, HIDDEN_DIM1, INPUT_DIM);
            add_bias<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_b1, current_batch_size, HIDDEN_DIM1);
            swish_forward<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_a1, current_batch_size * HIDDEN_DIM1);
            
            // 第二层: A1 * W2 + b2 -> Swish(h2)
            dim3 grid2((HIDDEN_DIM2 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid2, block>>>(d_a1, d_w2, d_h2, current_batch_size, HIDDEN_DIM2, HIDDEN_DIM1);
            add_bias<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_b2, current_batch_size, HIDDEN_DIM2);
            swish_forward<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_a2, current_batch_size * HIDDEN_DIM2);
            
            // 第三层: A2 * W3 + b3 -> Swish(h3)
            dim3 grid3((HIDDEN_DIM3 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid3, block>>>(d_a2, d_w3, d_h3, current_batch_size, HIDDEN_DIM3, HIDDEN_DIM2);
            add_bias<<<(current_batch_size * HIDDEN_DIM3 + 255) / 256, 256>>>(d_h3, d_b3, current_batch_size, HIDDEN_DIM3);
           swish_forward<<<(current_batch_size * HIDDEN_DIM3 + 255) / 256, 256>>>(d_h3, d_a3, current_batch_size * HIDDEN_DIM3);

            
            // 第四层: A3 * W4 + b4 -> output (线性输出层)
            dim3 grid4((OUTPUT_DIM + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid4, block>>>(d_a3, d_w4, d_output, current_batch_size, OUTPUT_DIM, HIDDEN_DIM3);
            add_bias<<<(current_batch_size * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_b4, current_batch_size, OUTPUT_DIM);
            
            // 计算损失
            hipMemset(d_loss, 0, sizeof(double));
            compute_mse_loss<<<(current_batch_size + 255) / 256, 256>>>(d_output, d_y, d_loss, current_batch_size);
            
            double batch_loss;
            hipMemcpy(&batch_loss, d_loss, sizeof(double), hipMemcpyDeviceToHost);
            batch_loss /= current_batch_size;
            total_loss += batch_loss;
            
            // 反向传播
            // 计算输出层梯度
            compute_output_grad<<<(current_batch_size + 255) / 256, 256>>>(d_output, d_y, d_grad_output, current_batch_size);
            
            // 计算第四层权重和偏置梯度
            transpose_matrix<<<((HIDDEN_DIM3 * current_batch_size) + 255) / 256, 256>>>(d_a3, d_temp1, current_batch_size, HIDDEN_DIM3);
            dim3 grid_w4((OUTPUT_DIM + 15) / 16, (HIDDEN_DIM3 + 15) / 16);
            matmul_optimized<<<grid_w4, block>>>(d_temp1, d_grad_output, d_grad_w4, HIDDEN_DIM3, OUTPUT_DIM, current_batch_size);
            compute_bias_grad<<<(OUTPUT_DIM + 255) / 256, 256>>>(d_grad_output, d_grad_b4, current_batch_size, OUTPUT_DIM);
            
            // 计算第三层隐藏层梯度
            transpose_matrix<<<((HIDDEN_DIM3 * OUTPUT_DIM) + 255) / 256, 256>>>(d_w4, d_temp2, HIDDEN_DIM3, OUTPUT_DIM);
            dim3 grid_h3((HIDDEN_DIM3 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid_h3, block>>>(d_grad_output, d_temp2, d_grad_h3, current_batch_size, HIDDEN_DIM3, OUTPUT_DIM);
            swish_backward<<<(current_batch_size * HIDDEN_DIM3 + 255) / 256, 256>>>(d_grad_h3, d_h3, d_a3, current_batch_size * HIDDEN_DIM3);
            
            // 计算第三层权重和偏置梯度
            transpose_matrix<<<((HIDDEN_DIM2 * current_batch_size) + 255) / 256, 256>>>(d_a2, d_temp1, current_batch_size, HIDDEN_DIM2);
            dim3 grid_w3((HIDDEN_DIM3 + 15) / 16, (HIDDEN_DIM2 + 15) / 16);
            matmul_optimized<<<grid_w3, block>>>(d_temp1, d_grad_h3, d_grad_w3, HIDDEN_DIM2, HIDDEN_DIM3, current_batch_size);
            compute_bias_grad<<<(HIDDEN_DIM3 + 255) / 256, 256>>>(d_grad_h3, d_grad_b3, current_batch_size, HIDDEN_DIM3);
            
            // 计算第二层隐藏层梯度
            transpose_matrix<<<((HIDDEN_DIM2 * HIDDEN_DIM3) + 255) / 256, 256>>>(d_w3, d_temp2, HIDDEN_DIM2, HIDDEN_DIM3);
            dim3 grid_h2((HIDDEN_DIM2 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid_h2, block>>>(d_grad_h3, d_temp2, d_grad_h2, current_batch_size, HIDDEN_DIM2, HIDDEN_DIM3);
            swish_backward<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_grad_h2, d_h2, d_a2, current_batch_size * HIDDEN_DIM2);
            
            // 计算第二层权重和偏置梯度
            transpose_matrix<<<((HIDDEN_DIM1 * current_batch_size) + 255) / 256, 256>>>(d_a1, d_temp1, current_batch_size, HIDDEN_DIM1);
            dim3 grid_w2((HIDDEN_DIM2 + 15) / 16, (HIDDEN_DIM1 + 15) / 16);
            matmul_optimized<<<grid_w2, block>>>(d_temp1, d_grad_h2, d_grad_w2, HIDDEN_DIM1, HIDDEN_DIM2, current_batch_size);
            compute_bias_grad<<<(HIDDEN_DIM2 + 255) / 256, 256>>>(d_grad_h2, d_grad_b2, current_batch_size, HIDDEN_DIM2);
            
            // 计算第一层隐藏层梯度
            transpose_matrix<<<((HIDDEN_DIM1 * HIDDEN_DIM2) + 255) / 256, 256>>>(d_w2, d_temp2, HIDDEN_DIM1, HIDDEN_DIM2);
            dim3 grid_h1((HIDDEN_DIM1 + 15) / 16, (current_batch_size + 15) / 16);
            matmul_optimized<<<grid_h1, block>>>(d_grad_h2, d_temp2, d_grad_h1, current_batch_size, HIDDEN_DIM1, HIDDEN_DIM2);
            swish_backward<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_grad_h1, d_h1, d_a1, current_batch_size * HIDDEN_DIM1);
            
            // 计算第一层权重和偏置梯度
            transpose_matrix<<<((INPUT_DIM * current_batch_size) + 255) / 256, 256>>>(d_X, d_temp1, current_batch_size, INPUT_DIM);
            dim3 grid_w1((HIDDEN_DIM1 + 15) / 16, (INPUT_DIM + 15) / 16);
            matmul_optimized<<<grid_w1, block>>>(d_temp1, d_grad_h1, d_grad_w1, INPUT_DIM, HIDDEN_DIM1, current_batch_size);
            compute_bias_grad<<<(HIDDEN_DIM1 + 255) / 256, 256>>>(d_grad_h1, d_grad_b1, current_batch_size, HIDDEN_DIM1);
            
            // Adam优化器更新参数
            int t = epoch * num_batches + batch_start / batch_samples + 1;
            
            adam_update_with_decay<<<(INPUT_DIM * HIDDEN_DIM1 + 255) / 256, 256>>>(
                d_w1, d_grad_w1, d_m_w1, d_v_w1, current_lr, beta1, beta2, eps, WEIGHT_DECAY, t, INPUT_DIM * HIDDEN_DIM1);
            adam_update_with_decay<<<(HIDDEN_DIM1 + 255) / 256, 256>>>(
                d_b1, d_grad_b1, d_m_w1, d_v_w1, current_lr, beta1, beta2, eps, 0.0, t, HIDDEN_DIM1);
                
            adam_update_with_decay<<<(HIDDEN_DIM1 * HIDDEN_DIM2 + 255) / 256, 256>>>(
                d_w2, d_grad_w2, d_m_w2, d_v_w2, current_lr, beta1, beta2, eps, WEIGHT_DECAY, t, HIDDEN_DIM1 * HIDDEN_DIM2);
            adam_update_with_decay<<<(HIDDEN_DIM2 + 255) / 256, 256>>>(
                d_b2, d_grad_b2, d_m_w2, d_v_w2, current_lr, beta1, beta2, eps, 0.0, t, HIDDEN_DIM2);
                
            adam_update_with_decay<<<(HIDDEN_DIM2 * HIDDEN_DIM3 + 255) / 256, 256>>>(
                d_w3, d_grad_w3, d_m_w3, d_v_w3, current_lr, beta1, beta2, eps, WEIGHT_DECAY, t, HIDDEN_DIM2 * HIDDEN_DIM3);
            adam_update_with_decay<<<(HIDDEN_DIM3 + 255) / 256, 256>>>(
                d_b3, d_grad_b3, d_m_w3, d_v_w3, current_lr, beta1, beta2, eps, 0.0, t, HIDDEN_DIM3);
                
            adam_update_with_decay<<<(HIDDEN_DIM3 * OUTPUT_DIM + 255) / 256, 256>>>(
                d_w4, d_grad_w4, d_m_w4, d_v_w4, current_lr, beta1, beta2, eps, WEIGHT_DECAY, t, HIDDEN_DIM3 * OUTPUT_DIM);
            adam_update_with_decay<<<(OUTPUT_DIM + 255) / 256, 256>>>(
                d_b4, d_grad_b4, d_m_w4, d_v_w4, current_lr, beta1, beta2, eps, 0.0, t, OUTPUT_DIM);
                
            hipDeviceSynchronize();
        }
        
        total_loss /= num_batches;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 早停和监控
        if (total_loss < best_loss * 0.999) {  // 改进阈值
            best_loss = total_loss;
            patience = 0;
        } else {
            patience++;
        }
        
        // 每50个epoch输出进度
        if (epoch % 50 == 0 || epoch == EPOCHS - 1) {
            std::cout << "[TRAIN] Epoch " << std::setw(4) << epoch 
                     << ", Loss: " << std::fixed << std::setprecision(6) << total_loss
                     << ", LR: " << std::scientific << std::setprecision(2) << current_lr
                     << ", Time: " << duration.count() << "ms" << std::endl;
        }
        
        // 早停
        if (patience >= max_patience) {
            std::cout << "[INFO] Early stopping triggered at epoch " << epoch << std::endl;
            break;
        }
    }
    
    // 训练完成后的评估
    hipMemcpy(h_w1.data(), d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b1.data(), d_b1, HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w2.data(), d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b2.data(), d_b2, HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w3.data(), d_w3, HIDDEN_DIM2 * HIDDEN_DIM3 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b3.data(), d_b3, HIDDEN_DIM3 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w4.data(), d_w4, HIDDEN_DIM3 * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b4.data(), d_b4, OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
    
    // 测试集评估
    std::cout << "\n[INFO] Evaluating on test set..." << std::endl;
    std::vector<double> predictions;
    int test_size = y_test.size();
    
    for (int i = 0; i < test_size; i += BATCH_SIZE) {
        int current_test_batch = std::min(BATCH_SIZE, test_size - i);
        
        hipMemcpy(d_X, X_test.data() + i * INPUT_DIM, 
                 current_test_batch * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        
        // 前向传播推理
        dim3 block(16, 16);
        
        // 第一层
        dim3 grid1((HIDDEN_DIM1 + 15) / 16, (current_test_batch + 15) / 16);
        matmul_optimized<<<grid1, block>>>(d_X, d_w1, d_h1, current_test_batch, HIDDEN_DIM1, INPUT_DIM);
        add_bias<<<(current_test_batch * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_b1, current_test_batch, HIDDEN_DIM1);
        swish_forward<<<(current_test_batch * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_a1, current_test_batch * HIDDEN_DIM1);
        
        // 第二层
        dim3 grid2((HIDDEN_DIM2 + 15) / 16, (current_test_batch + 15) / 16);
        matmul_optimized<<<grid2, block>>>(d_a1, d_w2, d_h2, current_test_batch, HIDDEN_DIM2, HIDDEN_DIM1);
        add_bias<<<(current_test_batch * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_b2, current_test_batch, HIDDEN_DIM2);
        swish_forward<<<(current_test_batch * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_a2, current_test_batch * HIDDEN_DIM2);
        
        // 第三层
        dim3 grid3((HIDDEN_DIM3 + 15) / 16, (current_test_batch + 15) / 16);
        matmul_optimized<<<grid3, block>>>(d_a2, d_w3, d_h3, current_test_batch, HIDDEN_DIM3, HIDDEN_DIM2);
        add_bias<<<(current_test_batch * HIDDEN_DIM3 + 255) / 256, 256>>>(d_h3, d_b3, current_test_batch, HIDDEN_DIM3);
        swish_forward<<<(current_test_batch * HIDDEN_DIM3 + 255) / 256, 256>>>(d_h3, d_a3, current_test_batch * HIDDEN_DIM3);
        
        // 第四层（输出层）
        dim3 grid4((OUTPUT_DIM + 15) / 16, (current_test_batch + 15) / 16);
        matmul_optimized<<<grid4, block>>>(d_a3, d_w4, d_output, current_test_batch, OUTPUT_DIM, HIDDEN_DIM3);
        add_bias<<<(current_test_batch * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_b4, current_test_batch, OUTPUT_DIM);
        
        hipDeviceSynchronize();
        
        // 复制预测结果到CPU
        std::vector<double> batch_pred(current_test_batch);
        hipMemcpy(batch_pred.data(), d_output, current_test_batch * sizeof(double), hipMemcpyDeviceToHost);
        predictions.insert(predictions.end(), batch_pred.begin(), batch_pred.end());
    }
    
    // 反归一化预测结果和真实值用于评估
    std::vector<double> pred_denorm = predictions;
    std::vector<double> test_denorm = y_test;
    denormalize_data_zscore(pred_denorm, mean_val, std_val);
    denormalize_data_zscore(test_denorm, mean_val, std_val);
    
    // 计算评估指标
    double mse, mae, r2;
    compute_metrics(pred_denorm, test_denorm, mse, mae, r2);
    
    std::cout << "\n=============== TEST RESULTS ===============" << std::endl;
    std::cout << "MSE:  " << std::fixed << std::setprecision(2) << mse << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(2) << sqrt(mse) << std::endl;
    std::cout << "MAE:  " << std::fixed << std::setprecision(2) << mae << std::endl;
    std::cout << "R²:   " << std::fixed << std::setprecision(4) << r2 << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // 显示部分预测结果
    std::cout << "\nSample predictions (first 10):" << std::endl;
    std::cout << "Predicted -> Actual" << std::endl;
    for (int i = 0; i < std::min(10, (int)pred_denorm.size()); i++) {
        std::cout << std::fixed << std::setprecision(2) 
                 << pred_denorm[i] << " -> " << test_denorm[i] << std::endl;
    }
    
    // 保存模型
    save_model_4layer(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3, h_w4, h_b4, 
                      mean_val, std_val, "4layer_bandwidth_model.txt");
    
    // 清理GPU内存
    hipFree(d_X); hipFree(d_y); hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
    hipFree(d_w3); hipFree(d_b3); hipFree(d_w4); hipFree(d_b4);
    hipFree(d_h1); hipFree(d_a1); hipFree(d_h2); hipFree(d_a2); hipFree(d_h3); hipFree(d_a3);
    hipFree(d_output); hipFree(d_loss);
    hipFree(d_grad_output); hipFree(d_grad_h3); hipFree(d_grad_h2); hipFree(d_grad_h1);
    hipFree(d_grad_w1); hipFree(d_grad_b1); hipFree(d_grad_w2); hipFree(d_grad_b2);
    hipFree(d_grad_w3); hipFree(d_grad_b3); hipFree(d_grad_w4); hipFree(d_grad_b4);
    hipFree(d_m_w1); hipFree(d_v_w1); hipFree(d_m_w2); hipFree(d_v_w2);
    hipFree(d_m_w3); hipFree(d_v_w3); hipFree(d_m_w4); hipFree(d_v_w4);
    hipFree(d_temp1); hipFree(d_temp2); hipFree(d_temp3);
    
    std::cout << "\n[INFO] Training completed successfully!" << std::endl;
    return 0;
}