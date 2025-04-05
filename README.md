# LModel-Optimization 🧠⚡

**Optimizing AI Models for Real-World Performance**

This project is focused on making AI models faster, smaller, and more efficient — especially for deployment in resource-constrained environments. The goal is to reduce inference time and memory footprint **without sacrificing accuracy**, making models practical for real-world use cases such as production APIs, mobile, and edge devices.

> 🔗 **Read the full article** for a deep dive into the theory, motivation, and implementation steps:  
[Optimizing AI Models for Real-World Performance: Tuning for Faster Inference](https://medium.com/@eddyejembi/optimizing-ai-models-for-real-world-performance-tuning-for-faster-inference-8d01b35b04d0)

---

## 🚀 What This Project Covers

This repo contains practical examples and code implementations for:

- **Quantization** — reducing model size and speeding up inference by converting weights from float32 to int8/float16.
- **Pruning** — removing unnecessary weights and neurons while maintaining performance.
- **Knowledge Distillation** — training smaller “student” models using outputs from a larger “teacher” model.
- **ONNX Export & Runtime Optimization** — converting models to ONNX format and using inference-optimized runtimes.
- **Benchmarking Tools** — to compare inference speed and model sizes before and after optimization.

---

## 🧩 Structure

```bash
LModel-Optimization/
├── base_bot.py
├── chatbot.py
├── load_model.py
├── optim_bot.py
├── optimum_optimize.py
└── requirements
```

## 🧪 Experimenting
### 📦 Installation
```
git clone https://github.com/EddyEjembi/LModel-Optimization.git
cd LModel-Optimization
pip install -r requirements.txt
```
### Running the Code
- **`load_model.py`** — Downloads the *`meta-llama/Llama-3.2-1B-Instruct`* model to serve as the base model.
- **`optimum_optimize.py`** — Optimizes and quantizes the model using HuggingFace Optimum.
- **`base_bot.py`** — Runs inference on the base model.
- **`optim_bot.py`** — Runs inference on the optimized model of your choice:

    ```
    # Load Models
    onnx_model_path = "onnx_model" # Directory for ONNX Model
    optimized_onnx_model_path = "optimized_onnx_model"  # Directory for Optimized ONNX model
    quantized_model_path = "quantized_onnx_model" # Directory to Quantized Model

    model = ORTModelForCausalLM.from_pretrained(optimized_onnx_model_path)
    ```

## 📚 Learn More
For a full walkthrough and real-world context, check out the article:

👉 [Optimizing AI Models for Real-World Performance](https://medium.com/@eddyejembi/optimizing-ai-models-for-real-world-performance-tuning-for-faster-inference-8d01b35b04d0)

## 🙌 Contributing
Pull requests are welcome! If you have better optimization techniques, benchmarks, or use cases — feel free to fork and contribute and connect with me on any of the platform:
- 🐦 [X (Twitter)](https://x.com/eddyejembi)
- 💼 [LinkedIn](https://www.linkedin.com/in/eddyejembi/)
- 📬 [Mail](mailto:eddyejembi2018@gmail.com)
