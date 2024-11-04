import torch
import time
import psutil
from transformers import AutoTokenizer


class Pirate:

    def __init__(self, model):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("<PATH-TO-TOKENIZER>")
        
        # Set the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Device Available: {device}")


        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.SYS_PROMPT = """
            You are an assistant for answering questions.
            Your name is Captain Jack Sparrow, and You are a Pirate.
            Respond to the User in a Pirate tone.
        """

    # Define a function for generating chatbot responses
    def generate(self, prompt):
        # Record the start time
        start_time = time.time()

        # Monitor resource usage before inference
        process = psutil.Process()  # current process
        cpu_usage_start = process.cpu_percent()
        mem_usage_start = process.memory_info().rss / (1024 ** 2)  # in MB

        # Prepare Input
        messages = [{"role":"system", "content":self.SYS_PROMPT}, {"role":"user", "content":prompt}]
        
        # tell the model to generate
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate attention mask
        attention_mask = torch.ones(input_ids.shape, device=self.model.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            eos_token_id=self.terminators[0],
            pad_token_id=self.terminators[0],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        # Calculate inference time
        inference_time = time.time() - start_time

        # Monitor resource usage after inference
        cpu_usage_end = process.cpu_percent()
        mem_usage_end = process.memory_info().rss / (1024 ** 2)  # in MB

        # Decode Response
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        # Print resource usage information
        print(f"Inference Time: {inference_time:.2f} seconds")
        print(f"CPU Usage: Start={cpu_usage_start}%, End={cpu_usage_end}%")
        print(f"Memory Usage: Start={mem_usage_start:.2f} MB, End={mem_usage_end:.2f} MB")
        print(f"Total Memory Usage: {mem_usage_end - mem_usage_start:.2f} MB")
        #print(f"Memory footprint: {self.model.get_memory_footprint() / 1e6:.2f} MB")

        return response
