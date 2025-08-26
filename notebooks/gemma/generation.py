def process_inputs(processor, image_url, prompt, device):
    messages = [
        {
            "role": "user", "content": [
                { "type": "image", "url": image_url },
                { "type": "text", "text": prompt },
            ]
        },
    ]

    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",    
        add_generation_prompt=True,
    ).to(device)
    

def generate(model, processor, inputs):
    with torch.inference_mode():
    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=200) 
    

    return processor.decode(outputs, skip_special_tokens=True)