import torch
from huggingface_hub import login
from transformers import Idefics3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainerCallback
import gc
from peft import LoraConfig, get_peft_model
import time
from trl import SFTConfig, SFTTrainer
import trackio
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dotenv import load_dotenv

SEED = 43
np.random.seed(SEED)

load_dotenv()
#HF_TOKEN = os.getenv("HF_TOKEN")
#login(HF_TOKEN)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class TrackioCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            trackio.log(logs)

def format_data(img: Image.Image, conf: float, validity_metric: float) -> dict:
    return {
      "images": [img],
      "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Predict the performance of the given object detection configuration on the provided image. Respond with a single numerical value representing the validity metric.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": f"Given the above image, our object detection model has a mean confidence score of {conf:.4f}. What is the expected validity metric for this configuration?",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": validity_metric
                }
            ],
        },
        ]
    }

def generate_text_from_sample(model: Idefics3ForConditionalGeneration, processor: AutoProcessor, sample: dict, max_new_tokens: int =1024, device: str ="cuda:0") -> str:
    clear_memory()
    
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = sample['images'][0]
    #image = sample[1]['content'][0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    clear_memory()
    return output_text[0]


def generate_text_from_batch(model, processor, batch_samples: list, max_new_tokens: int = 1024, device: str = "cuda:0") -> list[str]:
    text_inputs = []
    image_inputs = []

    # 1. Preprocess inputs into lists
    for sample in batch_samples:
        # Prepare text (Chat Template)
        # using [1:2] to grab the user message, skipping system prompt
        prompt = processor.apply_chat_template(
            sample['messages'][1:2], 
            add_generation_prompt=True
        )
        text_inputs.append(prompt)

        # Prepare image
        image = sample['images'][0]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Processor expects a list of lists for images (one list of images per sample)
        image_inputs.append([image])

    # 2. Tokenize and Pad Batch
    # padding=True is critical here to handle different sequence lengths in one batch
    model_inputs = processor(
        text=text_inputs,
        images=image_inputs,
        return_tensors="pt",
        padding=True 
    ).to(device)

    # 3. Generate
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # 4. Trim input tokens from output (Batched approach)
    # We zip input_ids and generated_ids to slice each row individually
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 5. Decode Batch
    output_texts = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_texts

    
def clear_memory(print_stats: bool = True) -> None:
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']

    time.sleep(2)
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    if print_stats:
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")



def test(test_dataset: list, validity_metric: str ="iou", batch_size: int = 4) -> None:
    clear_memory()
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    model = Idefics3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2",)
    processor = AutoProcessor.from_pretrained(model_id)
    adapter_path = f"models/assessors/smolvlm-{validity_metric}"
    model.load_adapter(adapter_path)

    y = []
    y_pred = []
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch_samples = test_dataset[i:i+batch_size]

        generated_texts = generate_text_from_batch(model, processor, batch_samples)
        for j, sample in enumerate(batch_samples):
            true_value = sample['messages'][-1]['content'][0]['text']
            gen_text = generated_texts[j] 
            try:
                if gen_text[-1] == '.':
                    gen_text = gen_text.strip('.')  

                generated_value = float(gen_text.strip())
                true_value_float = float(true_value)
                y.append(true_value_float)
                y_pred.append(generated_value)
            except ValueError:
                print(f"Could not convert generated text to float: {gen_text}")
                continue
        
        r_2 = r2_score(np.array(y), np.array(y_pred))
        mae = np.mean(np.abs(np.array(y) - np.array(y_pred)))
        mse = np.mean((np.array(y) - np.array(y_pred)) ** 2)
        print(f"Batch {i} Test R2: {r_2:.4f}")
        print(f"Batch {i} Test MAE: {mae:.4f}")
        print(f"Batch {i} Test MSE: {mse:.4f}")
    
        clear_memory(print_stats=False)
    y = np.array(y)
    y_pred = np.array(y_pred)
    r_2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))
    mse = np.mean((y - y_pred) ** 2)
    print(f"Test R2: {r_2:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    clear_memory()
    
    

def fine_tune(train_dataset: list, eval_dataset: list, validity_metric: str ="iou") -> None:
    model_id = "HuggingFaceTB/SmolVLM-Instruct"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) #load quant version 
    model = Idefics3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config, _attn_implementation="flash_attention_2",) # Load model and tokenizer
    processor = AutoProcessor.from_pretrained(model_id)

    peft_config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.1,target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'], use_dora=True, init_lora_weights="gaussian")  #qlora tuning config
    #peft_model = get_peft_model(model, peft_config)     # Apply PEFT model adaptation
    #print(peft_model.print_trainable_parameters())            


    training_args = SFTConfig(
        output_dir=f"models/assessors/smolvlm-{validity_metric}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        report_to="none",
        max_length=None
    )

    trackio.init(project=f"smolvlm-{validity_metric}", name=f"smolvlm-{validity_metric}", config=training_args, space_id=f"femartip/smolvlm-{validity_metric}-trackio")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=[TrackioCallback()]
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    trackio.finish()
    clear_memory()
    print("Fine-tuning completed and model saved.")


def get_system_data(data_path: str, target_metric: str = "iou") -> tuple[dict, dict]:
    data = pd.read_csv(data_path, index_col=0)
    conf_list = data["conf"].to_list()
    valid_metric_list = data[target_metric].to_list()
    image_ids = data.index.to_list()
    image_ids = ["{:06d}".format(int(img_id)) for img_id in image_ids]
    conf_dict = dict(zip(image_ids, conf_list))
    validity_metric_dict = dict(zip(image_ids, valid_metric_list))
    return conf_dict, validity_metric_dict

def get_image_dataset(image_folder: str, image_ids: list) -> dict:
    dataset = {}
    img_in_dir = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    for img_name in img_in_dir:
        img_id = img_name.split('_')[0]
        if img_id not in image_ids:
            continue
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path)
        image = image.resize((1333, 800))
        dataset[img_id] = image
    return dataset

if __name__ == "__main__":
    DATA_PATH = "data/data_yolo.csv"
    IMAGE_FOLDER = "data/zod_yolo/images/val"
    TARGET_METRIC = "iou"

    print("Loading model data and images.")
    conf, validity_metric = get_system_data(DATA_PATH, target_metric=TARGET_METRIC)
    img_ids = list(conf.keys())
    images = get_image_dataset(IMAGE_FOLDER, img_ids)
    print(f"Loaded {len(images)} images.")

    train_idx, test_idx = train_test_split(img_ids, test_size=0.4, train_size=0.6)
    train_idx, eval_idx = train_test_split(train_idx, test_size=0.2, train_size=0.8)

    train_dataset = [format_data(images[img_id], conf[img_id], validity_metric[img_id]) for img_id in train_idx]
    eval_dataset = [format_data(images[img_id], conf[img_id], validity_metric[img_id]) for img_id in eval_idx]
    test_dataset = [format_data(images[img_id], conf[img_id], validity_metric[img_id]) for img_id in test_idx]

    # TEST
    #model_id = "HuggingFaceTB/SmolVLM-Instruct"
    #model = Idefics3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2",)
    #processor = AutoProcessor.from_pretrained(model_id)
    #output = generate_text_from_sample(test_dataset[0])
    #print("Generated output:", output)

    #fine_tune(train_dataset, eval_dataset, validity_metric=TARGET_METRIC)
    test(test_dataset, validity_metric=TARGET_METRIC, batch_size=64)
    