/**
 * PINN Inference Engine - C++ ONNX Runtime Wrapper
 * High-performance inference for edge deployment.
 *
 * Input: float32[1, 3] = [x, y, aoa]
 * Output: float32[1, 3] = [u, v, p]
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
 *   make -j$(nproc)
 *
 * Usage:
 *   ./pinn_inference model.onnx [--benchmark N]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace pinn {

/**
 * PINN Inference Engine using ONNX Runtime
 */
class PINNInference {
public:
    /**
     * Constructor - loads ONNX model
     * @param model_path Path to ONNX model file
     * @param use_gpu Use CUDA execution provider if available
     * @param num_threads Number of threads for inference (0 = auto)
     */
    PINNInference(const std::string& model_path, bool use_gpu = false, int num_threads = 1)
        : env_(ORT_LOGGING_LEVEL_WARNING, "PINNInference"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Add CUDA provider if requested and available
        if (use_gpu) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[PINN] Using CUDA execution provider" << std::endl;
#else
            std::cout << "[PINN] CUDA not available, using CPU" << std::endl;
#endif
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_inputs = session_->GetInputCount();
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();
        
        auto input_shape_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        input_shape_ = input_shape_info.GetShape();
        
        // Output info
        size_t num_outputs = session_->GetOutputCount();
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name.get();
        
        auto output_shape_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        output_shape_ = output_shape_info.GetShape();
        
        std::cout << "[PINN] Model loaded: " << model_path << std::endl;
        std::cout << "[PINN] Input: " << input_name_ << " [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "[PINN] Output: " << output_name_ << " [";
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            std::cout << output_shape_[i];
            if (i < output_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    /**
     * Run inference on single input
     * @param x X-coordinate
     * @param y Y-coordinate
     * @param aoa Angle of attack (degrees or normalized)
     * @return Tuple of (u, v, p) predictions
     */
    std::tuple<float, float, float> predict(float x, float y, float aoa) {
        // Prepare input tensor
        std::vector<float> input_data = {x, y, aoa};
        std::vector<int64_t> input_shape = {1, 3};
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Extract output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        return std::make_tuple(output_data[0], output_data[1], output_data[2]);
    }
    
    /**
     * Run batch inference
     * @param inputs Vector of [x, y, aoa] inputs
     * @return Vector of [u, v, p] outputs
     */
    std::vector<std::vector<float>> predict_batch(const std::vector<std::vector<float>>& inputs) {
        size_t batch_size = inputs.size();
        
        // Flatten input data
        std::vector<float> input_data;
        input_data.reserve(batch_size * 3);
        for (const auto& input : inputs) {
            input_data.insert(input_data.end(), input.begin(), input.end());
        }
        
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), 3};
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Extract outputs
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        std::vector<std::vector<float>> outputs;
        outputs.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            outputs.push_back({
                output_data[i * 3 + 0],
                output_data[i * 3 + 1],
                output_data[i * 3 + 2]
            });
        }
        
        return outputs;
    }
    
    /**
     * Benchmark inference latency
     * @param n_iterations Number of iterations
     * @return Map of latency statistics
     */
    std::map<std::string, double> benchmark(int n_iterations = 1000) {
        std::vector<double> times;
        times.reserve(n_iterations);
        
        // Random input for benchmarking
        std::vector<float> input_data = {0.5f, 0.1f, 5.0f};
        std::vector<int64_t> input_shape = {1, 3};
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            predict(input_data[0], input_data[1], input_data[2]);
        }
        
        // Benchmark
        for (int i = 0; i < n_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            predict(input_data[0], input_data[1], input_data[2]);
            auto end = std::chrono::high_resolution_clock::now();
            
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(elapsed_ms);
        }
        
        // Calculate statistics
        std::sort(times.begin(), times.end());
        
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();
        
        double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / times.size() - mean * mean);
        
        size_t p50_idx = static_cast<size_t>(times.size() * 0.50);
        size_t p95_idx = static_cast<size_t>(times.size() * 0.95);
        size_t p99_idx = static_cast<size_t>(times.size() * 0.99);
        
        return {
            {"mean_ms", mean},
            {"std_ms", std_dev},
            {"min_ms", times.front()},
            {"max_ms", times.back()},
            {"p50_ms", times[p50_idx]},
            {"p95_ms", times[p95_idx]},
            {"p99_ms", times[p99_idx]}
        };
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};

} // namespace pinn


void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " MODEL_PATH [OPTIONS]\n"
              << "\n"
              << "Options:\n"
              << "  --benchmark N    Run N inference iterations for benchmarking\n"
              << "  --gpu            Use CUDA execution provider if available\n"
              << "  --threads N      Number of threads (default: 1)\n"
              << "  --test           Run test inference with sample input\n"
              << "  --help           Show this help message\n"
              << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string model_path;
    int benchmark_iterations = 0;
    bool use_gpu = false;
    int num_threads = 1;
    bool run_test = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--test") {
            run_test = true;
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: Model path required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        // Create inference engine
        pinn::PINNInference engine(model_path, use_gpu, num_threads);
        
        // Run test inference
        if (run_test || benchmark_iterations == 0) {
            std::cout << "\n[Test Inference]" << std::endl;
            
            float x = 0.5f, y = 0.1f, aoa = 5.0f;
            auto [u, v, p] = engine.predict(x, y, aoa);
            
            std::cout << "  Input:  x=" << x << ", y=" << y << ", aoa=" << aoa << std::endl;
            std::cout << "  Output: u=" << u << ", v=" << v << ", p=" << p << std::endl;
        }
        
        // Run benchmark
        if (benchmark_iterations > 0) {
            std::cout << "\n[Benchmark: " << benchmark_iterations << " iterations]" << std::endl;
            
            auto stats = engine.benchmark(benchmark_iterations);
            
            std::cout << "  Mean:  " << stats["mean_ms"] << " ms" << std::endl;
            std::cout << "  Std:   " << stats["std_ms"] << " ms" << std::endl;
            std::cout << "  Min:   " << stats["min_ms"] << " ms" << std::endl;
            std::cout << "  P50:   " << stats["p50_ms"] << " ms" << std::endl;
            std::cout << "  P95:   " << stats["p95_ms"] << " ms" << std::endl;
            std::cout << "  P99:   " << stats["p99_ms"] << " ms" << std::endl;
            
            if (stats["mean_ms"] < 1.0) {
                std::cout << "\n  ✓ Sub-millisecond inference achieved!" << std::endl;
            }
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
