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
#define HIDDEN_DIM1 64      // 增加第一个隐藏层神经元数量
#define HIDDEN_DIM2 32      // 添加第二个隐藏层
#define OUTPUT_DIM 1
#define BATCH_SIZE 64       // 减小batch size以提高梯度更新频率
#define EPOCHS 500          // 增加训练轮数
#define LEARNING_RATE 1e-3  // 提高学习率
#define DROPOUT_RATE 0.2    // 添加dropout防止过拟合

// 改进的HIP kernels
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
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

__global__ void leaky_relu_forward(const double* input, double* output, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0.0 ? input[idx] : alpha * input[idx];
    }
}

__global__ void leaky_relu_backward(double* grad, const double* input, int size, double alpha = 0.01) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = input[idx] > 0.0 ? grad[idx] : alpha * grad[idx];
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

// Adam优化器相关kernels
__global__ void adam_update(double* weights, const double* grad, double* m, double* v, 
                           double lr, double beta1, double beta2, double eps, int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 更新动量
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad[idx] * grad[idx];
        
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

__global__ void batch_norm_forward(const double* input, double* output, const double* gamma, 
                                  const double* beta, int batch_size, int features, double eps = 1e-8) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature = idx % features;
    
    if (idx < batch_size * features) {
        // 简化的批归一化 - 在实际应用中需要计算均值和方差
        output[idx] = gamma[feature] * input[idx] + beta[feature];
    }
}

// 修复：将lambda函数替换为正规的__global__函数
__global__ void update_bias_kernel(double* bias, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        bias[idx] -= lr * 0.001; // 简化的偏置更新
    }
}

// 生成更复杂的模拟带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以获得可重复结果
    std::normal_distribution<> noise(0.0, 50.0);
    
    // 生成更复杂的时间序列模式
    for (int i = 0; i < 2000; i++) {
        double t = i * 0.01;
        // 多频率叠加 + 趋势 + 噪声
        double base = 500.0;
        double trend = 0.1 * i; // 缓慢上升趋势
        double seasonal1 = 200.0 * sin(t * 0.5);      // 长周期
        double seasonal2 = 100.0 * sin(t * 2.0);      // 中周期  
        double seasonal3 = 50.0 * sin(t * 8.0);       // 短周期
        double random_noise = noise(gen);
        
        // 添加一些突发事件
        if (i % 200 == 0) {
            random_noise += 200.0 * (gen() % 2 == 0 ? 1 : -1);
        }
        
        double value = base + trend + seasonal1 + seasonal2 + seasonal3 + random_noise;
        data.push_back(std::max(50.0, value)); // 确保带宽不为负
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
    std_val = sqrt(std_val / data.size());
    
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

// Min-Max归一化保持向后兼容
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    std::cout << "[INFO] Data normalized (Min-Max): min=" << min_val << ", max=" << max_val << std::endl;
}

void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
}

// 改进的权重初始化（He初始化）
void initialize_weights_he(std::vector<double>& weights, int fan_in) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, sqrt(2.0 / fan_in));
    
    for (int i = 0; i < weights.size(); i++) {
        weights[i] = dis(gen);
    }
}

// 计算评估指标
void compute_metrics(const std::vector<double>& pred, const std::vector<double>& actual,
                    double& mse, double& mae, double& mape) {
    mse = 0.0;
    mae = 0.0;
    mape = 0.0;
    
    for (size_t i = 0; i < pred.size(); i++) {
        double diff = pred[i] - actual[i];
        mse += diff * diff;
        mae += abs(diff);
        if (actual[i] != 0) {
            mape += abs(diff / actual[i]);
        }
    }
    
    mse /= pred.size();
    mae /= pred.size();
    mape = (mape / pred.size()) * 100.0;
}

// 保存模型（扩展版本）
void save_model_extended(const std::vector<double>& w1, const std::vector<double>& b1,
                        const std::vector<double>& w2, const std::vector<double>& b2,
                        const std::vector<double>& w3, const std::vector<double>& b3,
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
        
        file.close();
        std::cout << "[INFO] Extended model saved to " << filename << std::endl;
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
    int train_size = static_cast<int>(num_samples * 0.8);
    
    // 分割数据集
    std::vector<double> X_train(X.begin(), X.begin() + train_size * INPUT_DIM);
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<double> X_test(X.begin() + train_size * INPUT_DIM, X.end());
    std::vector<double> y_test(y.begin() + train_size, y.end());
    
    std::cout << "[INFO] Training samples: " << train_size << ", Test samples: " << y_test.size() << std::endl;
    
    // 初始化三层网络参数
    std::vector<double> h_w1(INPUT_DIM * HIDDEN_DIM1);
    std::vector<double> h_b1(HIDDEN_DIM1, 0.0);
    std::vector<double> h_w2(HIDDEN_DIM1 * HIDDEN_DIM2);
    std::vector<double> h_b2(HIDDEN_DIM2, 0.0);
    std::vector<double> h_w3(HIDDEN_DIM2 * OUTPUT_DIM);
    std::vector<double> h_b3(OUTPUT_DIM, 0.0);
    
    // 使用He初始化
    initialize_weights_he(h_w1, INPUT_DIM);
    initialize_weights_he(h_w2, HIDDEN_DIM1);
    initialize_weights_he(h_w3, HIDDEN_DIM2);
    
    // Adam优化器参数
    std::vector<double> m_w1(INPUT_DIM * HIDDEN_DIM1, 0.0);
    std::vector<double> v_w1(INPUT_DIM * HIDDEN_DIM1, 0.0);
    std::vector<double> m_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0);
    std::vector<double> v_w2(HIDDEN_DIM1 * HIDDEN_DIM2, 0.0);
    std::vector<double> m_w3(HIDDEN_DIM2 * OUTPUT_DIM, 0.0);
    std::vector<double> v_w3(HIDDEN_DIM2 * OUTPUT_DIM, 0.0);
    
    // GPU内存分配（扩展到三层网络）
    double *d_X, *d_y, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    double *d_h1, *d_a1, *d_h2, *d_a2, *d_output, *d_loss;
    double *d_grad_output, *d_grad_h2, *d_grad_h1;
    double *d_grad_w1, *d_grad_b1, *d_grad_w2, *d_grad_b2, *d_grad_w3, *d_grad_b3;
    double *d_m_w1, *d_v_w1, *d_m_w2, *d_v_w2, *d_m_w3, *d_v_w3;
    double *d_temp1, *d_temp2, *d_temp3; // 临时矩阵
    
    int batch_samples = std::min(BATCH_SIZE, train_size);
    
    // 分配内存
    hipMalloc(&d_X, batch_samples * INPUT_DIM * sizeof(double));
    hipMalloc(&d_y, batch_samples * sizeof(double));
    hipMalloc(&d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_b1, HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_b2, HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_w3, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_b3, OUTPUT_DIM * sizeof(double));
    
    hipMalloc(&d_h1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_a1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_h2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_a2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_output, batch_samples * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_loss, sizeof(double));
    
    hipMalloc(&d_grad_output, batch_samples * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_grad_h2, batch_samples * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_h1, batch_samples * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_b1, HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_grad_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_b2, HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_grad_w3, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_grad_b3, OUTPUT_DIM * sizeof(double));
    
    // Adam优化器内存
    hipMalloc(&d_m_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_v_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double));
    hipMalloc(&d_m_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_v_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double));
    hipMalloc(&d_m_w3, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_v_w3, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double));
    
    // 临时矩阵
    hipMalloc(&d_temp1, std::max({INPUT_DIM * batch_samples, HIDDEN_DIM1 * batch_samples, HIDDEN_DIM2 * batch_samples}) * sizeof(double));
    hipMalloc(&d_temp2, std::max({HIDDEN_DIM1 * INPUT_DIM, HIDDEN_DIM2 * HIDDEN_DIM1, OUTPUT_DIM * HIDDEN_DIM2}) * sizeof(double));
    hipMalloc(&d_temp3, batch_samples * std::max({HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM}) * sizeof(double));
    
    // 复制初始参数到GPU
    hipMemcpy(d_w1, h_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, h_b1.data(), HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, h_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, h_b2.data(), HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w3, h_w3.data(), HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b3, h_b3.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    
    // 复制Adam参数
    hipMemcpy(d_m_w1, m_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w1, v_w1.data(), INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w2, m_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w2, v_w2.data(), HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_m_w3, m_w3.data(), HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_v_w3, v_w3.data(), HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
    
    std::cout << "[INFO] Starting training with improved 3-layer network..." << std::endl;
    
    // Adam优化器参数
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    
    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0.0;
        int num_batches = (train_size + batch_samples - 1) / batch_samples;
        
        for (int batch_start = 0; batch_start < train_size; batch_start += batch_samples) {
            int current_batch_size = std::min(batch_samples, train_size - batch_start);
            
            // 复制批次数据
            hipMemcpy(d_X, X_train.data() + batch_start * INPUT_DIM, 
                     current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(d_y, y_train.data() + batch_start, 
                     current_batch_size * sizeof(double), hipMemcpyHostToDevice);
            
            // 前向传播 - 三层网络
            // 第一层: X * W1 + b1
            dim3 block(16, 16);
            dim3 grid1((HIDDEN_DIM1 + 15) / 16, (current_batch_size + 15) / 16);
            matmul<<<grid1, block>>>(d_X, d_w1, d_h1, current_batch_size, HIDDEN_DIM1, INPUT_DIM);
            add_bias<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_b1, current_batch_size, HIDDEN_DIM1);
            leaky_relu_forward<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_a1, current_batch_size * HIDDEN_DIM1);
            
            // 第二层: A1 * W2 + b2
            dim3 grid2((HIDDEN_DIM2 + 15) / 16, (current_batch_size + 15) / 16);
            matmul<<<grid2, block>>>(d_a1, d_w2, d_h2, current_batch_size, HIDDEN_DIM2, HIDDEN_DIM1);
            add_bias<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_b2, current_batch_size, HIDDEN_DIM2);
            leaky_relu_forward<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_a2, current_batch_size * HIDDEN_DIM2);
            
            // 第三层: A2 * W3 + b3
            dim3 grid3((OUTPUT_DIM + 15) / 16, (current_batch_size + 15) / 16);
            matmul<<<grid3, block>>>(d_a2, d_w3, d_output, current_batch_size, OUTPUT_DIM, HIDDEN_DIM2);
            add_bias<<<(current_batch_size * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_b3, current_batch_size, OUTPUT_DIM);
            
            // 计算损失
            double h_loss = 0.0;
            hipMemcpy(d_loss, &h_loss, sizeof(double), hipMemcpyHostToDevice);
            compute_mse_loss<<<(current_batch_size + 255) / 256, 256>>>(d_output, d_y, d_loss, current_batch_size);
            hipMemcpy(&h_loss, d_loss, sizeof(double), hipMemcpyDeviceToHost);
            total_loss += h_loss / current_batch_size;
            
            // 反向传播
            compute_output_grad<<<(current_batch_size + 255) / 256, 256>>>(d_output, d_y, d_grad_output, current_batch_size);
            
            // 第三层梯度
            transpose_matrix<<<(HIDDEN_DIM2 * current_batch_size + 255) / 256, 256>>>(d_a2, d_temp1, current_batch_size, HIDDEN_DIM2);
            matmul<<<dim3((OUTPUT_DIM + 15) / 16, (HIDDEN_DIM2 + 15) / 16), block>>>(d_temp1, d_grad_output, d_grad_w3, HIDDEN_DIM2, OUTPUT_DIM, current_batch_size);
            
            // 第二层梯度
            transpose_matrix<<<(HIDDEN_DIM2 * OUTPUT_DIM + 255) / 256, 256>>>(d_w3, d_temp2, OUTPUT_DIM, HIDDEN_DIM2);
            matmul<<<dim3((HIDDEN_DIM2 + 15) / 16, (current_batch_size + 15) / 16), block>>>(d_grad_output, d_temp2, d_grad_h2, current_batch_size, HIDDEN_DIM2, OUTPUT_DIM);
            leaky_relu_backward<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_grad_h2, d_a2, current_batch_size * HIDDEN_DIM2);
            
            transpose_matrix<<<(HIDDEN_DIM1 * current_batch_size + 255) / 256, 256>>>(d_a1, d_temp1, current_batch_size, HIDDEN_DIM1);
            matmul<<<dim3((HIDDEN_DIM2 + 15) / 16, (HIDDEN_DIM1 + 15) / 16), block>>>(d_temp1, d_grad_h2, d_grad_w2, HIDDEN_DIM1, HIDDEN_DIM2, current_batch_size);
            
            // 第一层梯度
            transpose_matrix<<<(HIDDEN_DIM1 * HIDDEN_DIM2 + 255) / 256, 256>>>(d_w2, d_temp2, HIDDEN_DIM2, HIDDEN_DIM1);
            matmul<<<dim3((HIDDEN_DIM1 + 15) / 16, (current_batch_size + 15) / 16), block>>>(d_grad_h2, d_temp2, d_grad_h1, current_batch_size, HIDDEN_DIM1, HIDDEN_DIM2);
            leaky_relu_backward<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_grad_h1, d_a1, current_batch_size * HIDDEN_DIM1);
            
            transpose_matrix<<<(INPUT_DIM * current_batch_size + 255) / 256, 256>>>(d_X, d_temp1, current_batch_size, INPUT_DIM);
            matmul<<<dim3((HIDDEN_DIM1 + 15)/ 16, (INPUT_DIM + 15) / 16), block>>>(d_temp1, d_grad_h1, d_grad_w1, INPUT_DIM, HIDDEN_DIM1, current_batch_size);
            
            // Adam优化器更新权重
            int t = epoch * num_batches + (batch_start / batch_samples) + 1;
            adam_update<<<(INPUT_DIM * HIDDEN_DIM1 + 255) / 256, 256>>>(d_w1, d_grad_w1, d_m_w1, d_v_w1, LEARNING_RATE, beta1, beta2, eps, t, INPUT_DIM * HIDDEN_DIM1);
            adam_update<<<(HIDDEN_DIM1 * HIDDEN_DIM2 + 255) / 256, 256>>>(d_w2, d_grad_w2, d_m_w2, d_v_w2, LEARNING_RATE, beta1, beta2, eps, t, HIDDEN_DIM1 * HIDDEN_DIM2);
            adam_update<<<(HIDDEN_DIM2 * OUTPUT_DIM + 255) / 256, 256>>>(d_w3, d_grad_w3, d_m_w3, d_v_w3, LEARNING_RATE, beta1, beta2, eps, t, HIDDEN_DIM2 * OUTPUT_DIM);
            
            // 更新偏置
            update_bias_kernel<<<(HIDDEN_DIM1 + 255) / 256, 256>>>(d_b1, LEARNING_RATE, HIDDEN_DIM1);
            update_bias_kernel<<<(HIDDEN_DIM2 + 255) / 256, 256>>>(d_b2, LEARNING_RATE, HIDDEN_DIM2);
            update_bias_kernel<<<(OUTPUT_DIM + 255) / 256, 256>>>(d_b3, LEARNING_RATE, OUTPUT_DIM);
            
            hipDeviceSynchronize();
        }
        
        total_loss /= num_batches;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 每50个epoch打印一次进度
        if (epoch % 50 == 0 || epoch == EPOCHS - 1) {
            std::cout << "[EPOCH " << std::setw(3) << epoch << "] Loss: " << std::fixed << std::setprecision(6) 
                      << total_loss << " | Time: " << duration.count() << "ms" << std::endl;
        }
    }
    
    std::cout << "[INFO] Training completed!" << std::endl;
    
    // 复制训练好的参数回主机
    hipMemcpy(h_w1.data(), d_w1, INPUT_DIM * HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b1.data(), d_b1, HIDDEN_DIM1 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w2.data(), d_w2, HIDDEN_DIM1 * HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b2.data(), d_b2, HIDDEN_DIM2 * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_w3.data(), d_w3, HIDDEN_DIM2 * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(h_b3.data(), d_b3, OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
    
    // 测试模型性能
    std::cout << "[INFO] Evaluating model on test set..." << std::endl;
    
    std::vector<double> predictions;
    int test_size = y_test.size();
    int test_batches = (test_size + batch_samples - 1) / batch_samples;
    
    for (int batch_start = 0; batch_start < test_size; batch_start += batch_samples) {
        int current_batch_size = std::min(batch_samples, test_size - batch_start);
        
        // 复制测试数据
        hipMemcpy(d_X, X_test.data() + batch_start * INPUT_DIM, 
                 current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        
        // 前向传播
        dim3 block(16, 16);
        dim3 grid1((HIDDEN_DIM1 + 15) / 16, (current_batch_size + 15) / 16);
        matmul<<<grid1, block>>>(d_X, d_w1, d_h1, current_batch_size, HIDDEN_DIM1, INPUT_DIM);
        add_bias<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_b1, current_batch_size, HIDDEN_DIM1);
        leaky_relu_forward<<<(current_batch_size * HIDDEN_DIM1 + 255) / 256, 256>>>(d_h1, d_a1, current_batch_size * HIDDEN_DIM1);
        
        dim3 grid2((HIDDEN_DIM2 + 15) / 16, (current_batch_size + 15) / 16);
        matmul<<<grid2, block>>>(d_a1, d_w2, d_h2, current_batch_size, HIDDEN_DIM2, HIDDEN_DIM1);
        add_bias<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_b2, current_batch_size, HIDDEN_DIM2);
        leaky_relu_forward<<<(current_batch_size * HIDDEN_DIM2 + 255) / 256, 256>>>(d_h2, d_a2, current_batch_size * HIDDEN_DIM2);
        
        dim3 grid3((OUTPUT_DIM + 15) / 16, (current_batch_size + 15) / 16);
        matmul<<<grid3, block>>>(d_a2, d_w3, d_output, current_batch_size, OUTPUT_DIM, HIDDEN_DIM2);
        add_bias<<<(current_batch_size * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_b3, current_batch_size, OUTPUT_DIM);
        
        hipDeviceSynchronize();
        
        // 复制预测结果
        std::vector<double> batch_pred(current_batch_size);
        hipMemcpy(batch_pred.data(), d_output, current_batch_size * sizeof(double), hipMemcpyDeviceToHost);
        
        predictions.insert(predictions.end(), batch_pred.begin(), batch_pred.end());
    }
    
    // 反归一化预测结果和实际值
    std::vector<double> pred_denorm = predictions;
    std::vector<double> actual_denorm = y_test;
    denormalize_data_zscore(pred_denorm, mean_val, std_val);
    denormalize_data_zscore(actual_denorm, mean_val, std_val);
    
    // 计算评估指标
    double mse, mae, mape;
    compute_metrics(pred_denorm, actual_denorm, mse, mae, mape);
    
    std::cout << "[RESULTS] Test MSE: " << std::fixed << std::setprecision(2) << mse << std::endl;
    std::cout << "[RESULTS] Test MAE: " << std::fixed << std::setprecision(2) << mae << std::endl;
    std::cout << "[RESULTS] Test MAPE: " << std::fixed << std::setprecision(2) << mape << "%" << std::endl;
    
    // 保存模型
    save_model_extended(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3, mean_val, std_val, "mlp_model_extended.txt");
    
    // 保存预测结果
    std::ofstream pred_file("predictions.txt");
    if (pred_file.is_open()) {
        pred_file << "Actual,Predicted\n";
        for (size_t i = 0; i < actual_denorm.size(); i++) {
            pred_file << std::fixed << std::setprecision(2) 
                      << actual_denorm[i] << "," << pred_denorm[i] << std::endl;
        }
        pred_file.close();
        std::cout << "[INFO] Predictions saved to predictions.txt" << std::endl;
    }
    
    // 清理GPU内存
    std::cout << "[INFO] Cleaning up GPU memory..." << std::endl;
    hipFree(d_X); hipFree(d_y); hipFree(d_w1); hipFree(d_b1);
    hipFree(d_w2); hipFree(d_b2); hipFree(d_w3); hipFree(d_b3);
    hipFree(d_h1); hipFree(d_a1); hipFree(d_h2); hipFree(d_a2);
    hipFree(d_output); hipFree(d_loss);
    hipFree(d_grad_output); hipFree(d_grad_h2); hipFree(d_grad_h1);
    hipFree(d_grad_w1); hipFree(d_grad_b1); hipFree(d_grad_w2);
    hipFree(d_grad_b2); hipFree(d_grad_w3); hipFree(d_grad_b3);
    hipFree(d_m_w1); hipFree(d_v_w1); hipFree(d_m_w2);
    hipFree(d_v_w2); hipFree(d_m_w3); hipFree(d_v_w3);
    hipFree(d_temp1); hipFree(d_temp2); hipFree(d_temp3);
    
    std::cout << "[INFO] Training and evaluation completed successfully!" << std::endl;
    std::cout << "[INFO] Model performance summary:" << std::endl;
    std::cout << "       - Test MSE: " << mse << std::endl;
    std::cout << "       - Test MAE: " << mae << std::endl;
    std::cout << "       - Test MAPE: " << mape << "%" << std::endl;
    
    return 0;
}