"""
Deployment Packaging

Packages distilled, exported, and quantized models into deployment-ready
artifacts for specific target platforms:
- Web (ONNX.js / WebNN)
- Mobile (ONNX Runtime Mobile / PyTorch Mobile)
- Edge (C++ with ONNX Runtime)
- Server (FastAPI + Docker)
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PackageResult:
    """Result from packaging operation."""
    target: str
    output_dir: str
    files_created: list[str]
    metadata: dict[str, Any]


class DeploymentPackager:
    """
    Creates deployment-ready packages for various targets.
    
    Usage:
        packager = DeploymentPackager(
            model_path="model.onnx",
            metadata={"input_shape": [1, 4], ...}
        )
        result = packager.package_for_web(output_dir)
    """
    
    def __init__(
        self,
        model_path: str | Path,
        metadata: dict[str, Any] | None = None,
        model_name: str | None = None,
    ):
        """
        Args:
            model_path: Path to the ONNX model
            metadata: Model metadata (input/output specs, etc.)
            model_name: Name for the model in generated code
        """
        self.model_path = Path(model_path)
        self.metadata = metadata or {}
        self.model_name = model_name or self.model_path.stem
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def package_for_web(
        self,
        output_dir: str | Path,
        include_demo: bool = True,
        demo_type: str = "basic",
    ) -> PackageResult:
        """
        Create a web deployment package.
        
        Output:
            output_dir/
            ├── model.onnx
            ├── model_metadata.json
            ├── inference.js
            └── index.html (if include_demo)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # Copy model
        model_dest = output_dir / "model.onnx"
        shutil.copy2(self.model_path, model_dest)
        files_created.append(str(model_dest))
        
        # Write metadata
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        files_created.append(str(metadata_path))
        
        # Generate inference.js
        inference_js = self._generate_web_inference_js()
        inference_path = output_dir / "inference.js"
        with open(inference_path, 'w') as f:
            f.write(inference_js)
        files_created.append(str(inference_path))
        
        # Generate demo page if requested
        if include_demo:
            demo_html = self._generate_web_demo_html(demo_type)
            demo_path = output_dir / "index.html"
            with open(demo_path, 'w') as f:
                f.write(demo_html)
            files_created.append(str(demo_path))
        
        # Write README
        readme = self._generate_web_readme()
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        files_created.append(str(readme_path))
        
        logger.info(f"Created web package at {output_dir}")
        
        return PackageResult(
            target="web",
            output_dir=str(output_dir),
            files_created=files_created,
            metadata=self.metadata,
        )
    
    def package_for_mobile(
        self,
        output_dir: str | Path,
        framework: str = "onnx",
        include_example: bool = False,
    ) -> PackageResult:
        """
        Create a mobile deployment package.
        
        Output:
            output_dir/
            ├── model.onnx (or model.ort for optimized)
            ├── model_metadata.json
            └── inference_example.py
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # Copy/convert model
        if framework == "onnx":
            model_dest = output_dir / "model.onnx"
            shutil.copy2(self.model_path, model_dest)
            files_created.append(str(model_dest))
            
            # Try to create ORT format
            try:
                ort_path = self._convert_to_ort_format(output_dir)
                if ort_path:
                    files_created.append(str(ort_path))
            except Exception as e:
                logger.warning(f"Could not create ORT format: {e}")
        
        # Write metadata
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        files_created.append(str(metadata_path))
        
        # Generate example code
        if include_example:
            example_py = self._generate_mobile_example_py()
            example_path = output_dir / "inference_example.py"
            with open(example_path, 'w') as f:
                f.write(example_py)
            files_created.append(str(example_path))
        
        # Write README
        readme = self._generate_mobile_readme(framework)
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        files_created.append(str(readme_path))
        
        logger.info(f"Created mobile package at {output_dir}")
        
        return PackageResult(
            target="mobile",
            output_dir=str(output_dir),
            files_created=files_created,
            metadata=self.metadata,
        )
    
    def package_for_edge(
        self,
        output_dir: str | Path,
        target_platform: str = "arm64",
        include_cpp: bool = True,
        include_python: bool = True,
        include_sample_data: bool = False,
    ) -> PackageResult:
        """
        Create an edge deployment package with C++ inference code.
        
        Output:
            output_dir/
            ├── model.onnx
            ├── model_metadata.json
            ├── CMakeLists.txt
            ├── inference.cpp
            ├── inference.py
            └── README.md
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        # Copy model
        model_dest = output_dir / "model.onnx"
        shutil.copy2(self.model_path, model_dest)
        files_created.append(str(model_dest))
        
        # Write metadata
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        files_created.append(str(metadata_path))
        
        # Generate C++ inference code
        if include_cpp:
            cpp_code = self._generate_cpp_inference()
            cpp_path = output_dir / "inference.cpp"
            with open(cpp_path, 'w') as f:
                f.write(cpp_code)
            files_created.append(str(cpp_path))
            
            cmake = self._generate_cmake(target_platform)
            cmake_path = output_dir / "CMakeLists.txt"
            with open(cmake_path, 'w') as f:
                f.write(cmake)
            files_created.append(str(cmake_path))
        
        # Generate Python inference script
        if include_python:
            py_code = self._generate_edge_python()
            py_path = output_dir / "inference.py"
            with open(py_path, 'w') as f:
                f.write(py_code)
            files_created.append(str(py_path))
        
        # Generate sample data
        if include_sample_data:
            sample_path = output_dir / "sample_input.json"
            input_shape = self.metadata.get('input_shape', [1, 4])
            import numpy as np
            sample_data = np.random.randn(*input_shape).tolist()
            with open(sample_path, 'w') as f:
                json.dump({"input": sample_data}, f)
            files_created.append(str(sample_path))
        
        # Write README
        readme = self._generate_edge_readme(target_platform)
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        files_created.append(str(readme_path))
        
        logger.info(f"Created edge package at {output_dir}")
        
        return PackageResult(
            target="edge",
            output_dir=str(output_dir),
            files_created=files_created,
            metadata=self.metadata,
        )
    
    def _convert_to_ort_format(self, output_dir: Path) -> Path | None:
        """Convert ONNX model to ORT format for mobile."""
        try:
            import onnxruntime as ort
            
            # ORT format is created by optimizing the model
            # This is a simplified version
            ort_path = output_dir / "model.ort"
            
            # For now, just copy the ONNX model
            # Real implementation would use onnxruntime_tools
            return None
            
        except ImportError:
            return None
    
    def _generate_web_inference_js(self) -> str:
        """Generate JavaScript inference code for ONNX.js."""
        input_shape = self.metadata.get('input_shape', [1, 4])
        output_shape = self.metadata.get('output_shape', [1, 1])
        
        return f'''/**
 * {self.model_name} - ONNX.js Inference
 * 
 * Auto-generated inference code for web deployment.
 */

import * as ort from 'onnxruntime-web';

// Model configuration
const MODEL_PATH = './model.onnx';
const INPUT_SHAPE = {input_shape};
const OUTPUT_SHAPE = {output_shape};

let session = null;

/**
 * Initialize the ONNX Runtime session.
 * Call this once before running inference.
 */
export async function initModel() {{
    if (session !== null) {{
        return;
    }}
    
    try {{
        session = await ort.InferenceSession.create(MODEL_PATH, {{
            executionProviders: ['webgl', 'wasm'],
            graphOptimizationLevel: 'all',
        }});
        console.log('Model loaded successfully');
    }} catch (error) {{
        console.error('Failed to load model:', error);
        throw error;
    }}
}}

/**
 * Run inference on input data.
 * 
 * @param {{Float32Array|number[]}} inputData - Input data as flat array
 * @returns {{Promise<Float32Array>}} Model output
 */
export async function predict(inputData) {{
    if (session === null) {{
        await initModel();
    }}
    
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', 
        Float32Array.from(inputData), 
        INPUT_SHAPE
    );
    
    // Run inference
    const feeds = {{ input: inputTensor }};
    const results = await session.run(feeds);
    
    // Get output
    const output = results.output;
    return output.data;
}}

/**
 * Run batch inference.
 * 
 * @param {{number[][]}} batchData - Array of input arrays
 * @returns {{Promise<number[][]>}} Array of outputs
 */
export async function predictBatch(batchData) {{
    const results = [];
    for (const input of batchData) {{
        const output = await predict(input);
        results.push(Array.from(output));
    }}
    return results;
}}

// Export for use as ES module or script
if (typeof window !== 'undefined') {{
    window.{self.model_name.replace('-', '_')} = {{
        initModel,
        predict,
        predictBatch,
    }};
}}
'''
    
    def _generate_web_demo_html(self, demo_type: str) -> str:
        """Generate HTML demo page."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.model_name} Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        textarea {{
            width: 100%;
            height: 100px;
            font-family: monospace;
            padding: 10px;
            margin: 10px 0;
        }}
        button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{ background: #0056b3; }}
        button:disabled {{ background: #ccc; cursor: not-allowed; }}
        #output {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
            white-space: pre-wrap;
            font-family: monospace;
        }}
        .loading {{ color: #666; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.model_name}</h1>
        <p>Enter input values (comma-separated):</p>
        <textarea id="input" placeholder="0.1, 0.2, 0.3, 0.4">{', '.join(['0.0'] * self.metadata.get('input_shape', [1, 4])[-1])}</textarea>
        <button id="runBtn" onclick="runInference()">Run Inference</button>
        <div id="output"></div>
    </div>

    <script>
        let session = null;
        
        async function initModel() {{
            const outputDiv = document.getElementById('output');
            outputDiv.innerHTML = '<span class="loading">Loading model...</span>';
            
            try {{
                session = await ort.InferenceSession.create('./model.onnx');
                outputDiv.innerHTML = 'Model loaded. Ready for inference.';
            }} catch (error) {{
                outputDiv.innerHTML = 'Error loading model: ' + error.message;
            }}
        }}
        
        async function runInference() {{
            if (!session) {{
                await initModel();
            }}
            
            const inputText = document.getElementById('input').value;
            const outputDiv = document.getElementById('output');
            const runBtn = document.getElementById('runBtn');
            
            try {{
                runBtn.disabled = true;
                outputDiv.innerHTML = '<span class="loading">Running inference...</span>';
                
                // Parse input
                const inputData = inputText.split(',').map(x => parseFloat(x.trim()));
                
                // Create tensor
                const tensor = new ort.Tensor('float32', 
                    Float32Array.from(inputData), 
                    [1, inputData.length]
                );
                
                // Run inference
                const startTime = performance.now();
                const results = await session.run({{ input: tensor }});
                const endTime = performance.now();
                
                // Display results
                const output = results.output.data;
                outputDiv.innerHTML = 
                    'Output: [' + Array.from(output).map(x => x.toFixed(6)).join(', ') + ']\\n' +
                    'Inference time: ' + (endTime - startTime).toFixed(2) + ' ms';
                    
            }} catch (error) {{
                outputDiv.innerHTML = 'Error: ' + error.message;
            }} finally {{
                runBtn.disabled = false;
            }}
        }}
        
        // Initialize on page load
        initModel();
    </script>
</body>
</html>
'''
    
    def _generate_web_readme(self) -> str:
        """Generate README for web package."""
        return f'''# {self.model_name} - Web Deployment

## Quick Start

1. Serve this directory with a web server:
   ```bash
   python -m http.server 8000
   # or
   npx serve .
   ```

2. Open `http://localhost:8000` in your browser

## Usage in Your Project

### ES Module

```javascript
import {{ initModel, predict }} from './inference.js';

await initModel();
const result = await predict([0.1, 0.2, 0.3, 0.4]);
console.log(result);
```

### Script Tag

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
<script type="module" src="inference.js"></script>
<script>
    async function run() {{
        await {self.model_name.replace('-', '_')}.initModel();
        const result = await {self.model_name.replace('-', '_')}.predict([0.1, 0.2, 0.3, 0.4]);
        console.log(result);
    }}
    run();
</script>
```

## Files

- `model.onnx` - ONNX model
- `model_metadata.json` - Model input/output specification
- `inference.js` - JavaScript inference wrapper
- `index.html` - Demo page

## Requirements

- Modern browser with WebAssembly support
- ONNX Runtime Web (loaded from CDN)
'''
    
    def _generate_mobile_example_py(self) -> str:
        """Generate Python example for mobile."""
        return f'''"""
{self.model_name} - Mobile Inference Example

Example showing how to run inference with ONNX Runtime Mobile.
"""

import numpy as np
import onnxruntime as ort

def load_model(model_path: str = "model.onnx"):
    """Load the ONNX model."""
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return session

def predict(session, input_data: np.ndarray) -> np.ndarray:
    """Run inference."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {{input_name: input_data.astype(np.float32)}})
    return output[0]

if __name__ == "__main__":
    # Load model
    session = load_model()
    
    # Create sample input
    input_shape = {self.metadata.get('input_shape', [1, 4])}
    sample_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    result = predict(session, sample_input)
    print(f"Input shape: {{sample_input.shape}}")
    print(f"Output shape: {{result.shape}}")
    print(f"Output: {{result}}")
'''
    
    def _generate_mobile_readme(self, framework: str) -> str:
        """Generate README for mobile package."""
        return f'''# {self.model_name} - Mobile Deployment

## Requirements

```bash
pip install onnxruntime  # or onnxruntime-mobile for smaller footprint
```

## Usage

```python
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession("model.onnx")

# Prepare input
input_data = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {{input_name: input_data}})

print(output[0])
```

## Files

- `model.onnx` - ONNX model (standard format)
- `model.ort` - ORT format (optimized for mobile, if available)
- `model_metadata.json` - Model specification
'''
    
    def _generate_cpp_inference(self) -> str:
        """Generate C++ inference code."""
        input_dim = self.metadata.get('input_shape', [1, 4])[-1]
        output_dim = self.metadata.get('output_shape', [1, 1])[-1]
        
        return f'''/**
 * {self.model_name} - C++ ONNX Runtime Inference
 * 
 * Compile with:
 *   mkdir build && cd build
 *   cmake .. && make
 * 
 * Run:
 *   ./inference ../model.onnx
 */

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

class ModelInference {{
public:
    ModelInference(const char* model_path) {{
        // Create environment
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Create session
        session_ = Ort::Session(env_, model_path, session_options);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    }}
    
    std::vector<float> predict(const std::vector<float>& input) {{
        // Create input tensor
        std::vector<int64_t> input_shape = {{1, {input_dim}}};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {{input_name_.c_str()}};
        const char* output_names[] = {{output_name_.c_str()}};
        
        auto output_tensors = session_.Run(
            Ort::RunOptions{{nullptr}},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );
        
        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;
        
        return std::vector<float>(output_data, output_data + output_size);
    }}
    
    double benchmark(int num_runs = 1000, int warmup = 50) {{
        std::vector<float> input({input_dim}, 0.0f);
        
        // Warmup
        for (int i = 0; i < warmup; i++) {{
            predict(input);
        }}
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {{
            predict(input);
        }}
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0 / num_runs;  // ms per inference
    }}

private:
    Ort::Env env_{{nullptr}};
    Ort::Session session_{{nullptr}};
    std::string input_name_;
    std::string output_name_;
}};

int main(int argc, char* argv[]) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }}
    
    try {{
        ModelInference model(argv[1]);
        
        // Example inference
        std::vector<float> input({input_dim}, 0.1f);
        auto output = model.predict(input);
        
        std::cout << "Output: [";
        for (size_t i = 0; i < output.size(); i++) {{
            std::cout << output[i];
            if (i < output.size() - 1) std::cout << ", ";
        }}
        std::cout << "]" << std::endl;
        
        // Benchmark
        double latency = model.benchmark();
        std::cout << "Latency: " << latency << " ms" << std::endl;
        
    }} catch (const Ort::Exception& e) {{
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }}
    
    return 0;
}}
'''
    
    def _generate_cmake(self, target_platform: str) -> str:
        """Generate CMakeLists.txt."""
        return f'''cmake_minimum_required(VERSION 3.14)
project({self.model_name.replace('-', '_')}_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find ONNX Runtime
# Option 1: System installed
find_package(onnxruntime QUIET)

# Option 2: Manual path (set ONNXRUNTIME_ROOT_DIR)
if(NOT onnxruntime_FOUND)
    if(DEFINED ENV{{ONNXRUNTIME_ROOT_DIR}})
        set(ONNXRUNTIME_ROOT_DIR $ENV{{ONNXRUNTIME_ROOT_DIR}})
    endif()
    
    if(ONNXRUNTIME_ROOT_DIR)
        include_directories(${{ONNXRUNTIME_ROOT_DIR}}/include)
        link_directories(${{ONNXRUNTIME_ROOT_DIR}}/lib)
        set(ONNXRUNTIME_LIBS onnxruntime)
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT_DIR.")
    endif()
else()
    set(ONNXRUNTIME_LIBS onnxruntime::onnxruntime)
endif()

# Inference executable
add_executable(inference inference.cpp)
target_link_libraries(inference ${{ONNXRUNTIME_LIBS}})

# Platform-specific settings
if("{target_platform}" STREQUAL "arm64")
    # ARM optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(inference PRIVATE -march=armv8-a)
    endif()
elseif("{target_platform}" STREQUAL "x86_64")
    # x86-64 optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(inference PRIVATE -march=native)
    endif()
endif()

# Install
install(TARGETS inference DESTINATION bin)
install(FILES model.onnx DESTINATION share/{self.model_name})
'''
    
    def _generate_edge_python(self) -> str:
        """Generate Python edge inference script."""
        return f'''#!/usr/bin/env python3
"""
{self.model_name} - Edge Inference

Simple inference script for edge deployment.
"""

import argparse
import json
import time
import numpy as np

def load_model(model_path: str):
    """Load ONNX model."""
    import onnxruntime as ort
    return ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )

def predict(session, input_data: np.ndarray) -> np.ndarray:
    """Run inference."""
    input_name = session.get_inputs()[0].name
    return session.run(None, {{input_name: input_data.astype(np.float32)}})[0]

def benchmark(session, input_shape, num_runs: int = 1000):
    """Benchmark inference latency."""
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(50):
        predict(session, input_data)
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        predict(session, input_data)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)
    
    return {{
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }}

def main():
    parser = argparse.ArgumentParser(description='{self.model_name} inference')
    parser.add_argument('--model', default='model.onnx', help='Model path')
    parser.add_argument('--input', help='Input JSON file')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    args = parser.parse_args()
    
    # Load metadata
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    input_shape = metadata.get('input_shape', [1, 4])
    
    # Load model
    session = load_model(args.model)
    
    if args.benchmark:
        results = benchmark(session, input_shape)
        print(f"Latency: {{results['mean_ms']:.2f}} ms (mean)")
        print(f"         {{results['p95_ms']:.2f}} ms (p95)")
        print(f"         {{results['p99_ms']:.2f}} ms (p99)")
    elif args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
        input_data = np.array(data['input'], dtype=np.float32)
        output = predict(session, input_data)
        print(f"Output: {{output.tolist()}}")
    else:
        # Demo with random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        output = predict(session, input_data)
        print(f"Input: {{input_data.tolist()}}")
        print(f"Output: {{output.tolist()}}")

if __name__ == '__main__':
    main()
'''
    
    def _generate_edge_readme(self, target_platform: str) -> str:
        """Generate README for edge package."""
        return f'''# {self.model_name} - Edge Deployment

## Python Usage

```bash
# Install dependencies
pip install onnxruntime numpy

# Run inference
python inference.py

# Benchmark
python inference.py --benchmark

# With input file
python inference.py --input sample_input.json
```

## C++ Build

### Prerequisites

1. Install ONNX Runtime:
   - Download from https://github.com/microsoft/onnxruntime/releases
   - Set `ONNXRUNTIME_ROOT_DIR` environment variable

### Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Run

```bash
./inference ../model.onnx
```

## Target Platform: {target_platform}

This package is optimized for {target_platform} deployment.

## Files

- `model.onnx` - ONNX model
- `model_metadata.json` - Model specification
- `inference.cpp` - C++ inference code
- `inference.py` - Python inference script
- `CMakeLists.txt` - CMake build configuration
'''
