from optimum.onnxruntime import ORTModelForCausalLM, ORTOptimizer, ORTQuantizer, AutoOptimizationConfig
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model_name = "C:/Users/Eddy Ejembi/Documents/MODELS/llama3.2-1B/MODEL"  # Replace with your actual model path/name

onnx_model_path = "onnx_model" # Directory for ONNX Model
optimized_onnx_model_path = "optimized_onnx_model"  # Directory for Optimized ONNX model
quantized_model_path = "quantized_onnx_model" # Directory to Quantized Model

# Convert Model to ONNX
optimum_model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
optimum_model.save_pretrained(onnx_model_path) # Save ONNX Model


# GRAPH OPTIMIZATION
optimizer = ORTOptimizer.from_pretrained(optimum_model)
optimization_config = AutoOptimizationConfig.O2()  # Use AutoOptimizationConfig.02() for dynamic optimization
# Optimize the model
optimizer.optimize(save_dir=optimized_onnx_model_path, optimization_config=optimization_config)


# QUANTIZATION
quantizer = ORTQuantizer.from_pretrained(optimum_model)
quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)  # Use AutoQuantizationConfig.[system] for dynamic quantization
# Apply quantization
quantizer.quantize(save_dir=quantized_model_path, quantization_config=quantization_config)
