from optimum.onnxruntime import ORTModelForCausalLM
from chatbot import Pirate

# Load Models
onnx_model_path = "onnx_model" # Directory for ONNX Model
optimized_onnx_model_path = "optimized_onnx_model"  # Directory for Optimized ONNX model
quantized_model_path = "quantized_onnx_model" # Directory to Quantized Model

model = ORTModelForCausalLM.from_pretrained(optimized_onnx_model_path)

# Initialize the Pirate Class
pirate = Pirate(model)

# Test the chatbot
while True:
    input_text = input("Enter your Message: ")
    response = pirate.generate(input_text)
    print(f"Jack Sparrow 🏴‍☠️: {response}")
