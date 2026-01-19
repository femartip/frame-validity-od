from string import Template
import os
import json
import pandas as pd
import re
import json
from dotenv import dotenv_values
from tqdm import tqdm

import base64
from tenacity import retry as tenacity_retry, wait_exponential, stop_after_delay, RetryError

SEED = 42

import openai
from openai import OpenAI 
openai.api_key = dotenv_values(".env")["OPENAI_KEY"]

from google import genai
from google.genai import types
GEMINI_API_KEY = dotenv_values(".env")["GOOGLE_API_KEY"]


init_prompt = Template("""
{
    "system_message": "IMPORTANT: Return only a valid JSON object with no explanations, text, or markdown!!! Do not include any commentary or introductory text!!!",
    "input_metadata": {
        "dataset_name": "$name",
        "description": "$description",
        "examples":
    """)

final_prompt = """
},
    "task": {
        "steps": [
            "Analyze the provided metadata and examples to determine the domain and context of the dataset.",
            "Identify the key characteristics of the dataset relevant to predicting the target variable.",
            "List potential high-level categorical and numerical features based on domain knowledge inferred from the dataset description.",
            "Extract additional potential features from dataset examples using syntactic and semantic patterns, ensuring at least 20 distinct features are generated.",
            "If the text implies certain values that match the target, these values may also be extracted as features. In cases where the target has multiple values, each value can be independently derived from the text as a feature if it is contextually appropriate.",
            "For text-based datasets, identify key phrases, structural components, and linguistic patterns that are relevant.",
            "For numerical datasets, identify aggregation patterns, distributional characteristics, and possible transformations.",
            "Group related features into meaningful categories where applicable.",
            "If a feature has more than 15 unique categories, group less frequent categories into an 'Other' class.",
            "For each identified feature, provide a clear name, description, a complete list of possible values, and a specific LLM extraction query."
        ],
        "constraints": [
            "Ensure features are distinct and non-redundant.",
            "Note that the target variable is not explicitly present in the input text.",
            "Prioritize domain-specific insights over generic ones.",
            "Ensure output is a structured, valid JSON format.",
            "For categorical variables, list possible values with domain justification.",
            "For numeric variables, provide possible transformations (e.g., log, mean differences).",
            "The extraction queries must be specific and detailed to ensure high-quality feature generation.",
            "Tailor extraction queries to the domain context of the dataset.",
            "Generate a diverse set of features to maximize potential predictive power."
        ]
    },
    "output_format": {
        "type": "json",
        "structure": {
            "features": [
                {
                    "feature_name": "<Name of the categorical or numerical feature>",
                    "description": "<Short description of what the feature represents and how it relates to the dataset's context>",
                    "possible_values": ["<Value 1>", "<Value 2>", "...", "<Value n>"],
                    "extraction_query": "Identify the '<feature_name>' based on the provided context. Options: '<Value 1>', '<Value 2>', ..., '<Value n>'."
                }
            ]
        }
    }
}
"""


def generate_feature_query(features: str, image_path: str, llm: str) -> list:
    with open(image_path, "rb") as image_file:
        img = image_file.read()
        img_b64 = base64.b64encode(img).decode('utf-8')
    query = []
    if llm == "gemini":
        query.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))        
        query.append("""
            "task": "Extract the following features as described below and return a valid JSON object.",
            "constraints": [
                "The output must be a valid JSON.",
                "All answers must be simple and correspond to categorical values only."
            ],
            "features": [
        """
        + features +              
        """
            ],
            "output_format": {
                "type": "json",
                "structure": {
                    "features": [
                        {
                            "feature_name": "<Feature Name>",
                            "answer": "<Extracted Answer>"
                        }
                    ]
                }
            }""")
    
    elif llm == "openai":
        query.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})        
        query.append({"type": "text", "text": """
            "task": "Extract the following features as described below and return a valid JSON object.",
            "constraints": [
                "The output must be a valid JSON.",
                "All answers must be simple and correspond to categorical values only."
            ],
            "features": [
        """
        + features +              
        """
            ],
            "output_format": {
                "type": "json",
                "structure": {
                    "features": [
                        {
                            "feature_name": "<Feature Name>",
                            "answer": "<Extracted Answer>"
                        }
                    ]
                }
            }"""})
    
    return query


# Generate prompt
def generate_prompt(name: str, description: str, df: pd.DataFrame, image_path: str, img_file_names: list, llm: str) -> list[dict]:
    RAND_N_IMG = 10
    random_img_id = df.sample(n=RAND_N_IMG, random_state=SEED).index.tolist()
    imgs = [] 
    imgs_bs64 = []
    for img_id in random_img_id:
        img_file_name = img_file_names[img_id]
        img_path_name = image_path + img_file_name
        with open(img_path_name, "rb") as image_file:
            img = image_file.read()
            img_b64 = base64.b64encode(img).decode('utf-8')
        imgs.append(img)
        imgs_bs64.append(img_b64)
    
    prompt = []
    if llm == "gemini":
        prompt.append(init_prompt.substitute(name=name, description=description))
        prompt.extend([types.Part.from_bytes(data=data_byte, mime_type="image/jpeg") for data_byte in imgs])
        prompt.append(final_prompt)

    elif llm == "openai":
        prompt.append({"type": "text", "text": init_prompt.substitute(name=name, description=description)})
        prompt.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data_b64}"}} for data_b64 in imgs_bs64])
        prompt.append({"type": "text", "text": final_prompt})

    return prompt

def get_files_by_ids(img_ids: list, img_file_names) -> dict:
    file_names = {}
    for img_id in img_ids:
        img_id_str = str(img_id).zfill(6)
        pattern = re.compile(rf"\b{img_id_str}_.*\.jpg")
        img_file_name = list(filter(pattern.search, img_file_names))        
        
        if len(img_file_name) != 1:
            print("WARNING: Image not found.")
        else:
            file_names[img_id] = img_file_name[0]

    return file_names


def extract_json(response: str) -> dict:
    # Find JSON between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON Error: {e}")
    else:
        return json.loads(response.strip('```json').strip('```'))


@tenacity_retry(
    wait=wait_exponential(multiplier=1, min=10, max=60),
    stop=stop_after_delay(800),
    reraise=True,
)
def get_response(llm: str, prompt: list) -> str:
    if llm == "gemini":
        # Google config
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        return str(response.text)

    elif llm == "openai":    
        # OpenAI conifg
        client = OpenAI(api_key=dotenv_values(".env")["OPENAI_KEY"],) 
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[
                {"role": "user", "content": prompt}            #type: ignore
            ],
            )
        return str(response.choices[0].message.content)
    else:
        raise ValueError()
    

def feature_discovery(llm: str, dataset: dict) -> str:
    # Feature discovery
    json_prompt = generate_prompt(dataset['name'], dataset['description'], dataset['df'], dataset['image_path'], dataset['img_file_names'], llm)
    feature_str = get_response(llm, json_prompt)
    json_features = extract_json(feature_str)
    num_features = len(json_features["features"])
    print(f"Features extracted information: \nNum of features {num_features} \nFeatures{feature_str}")
    with open("./data/llm_metafeatures_description.json", "w+") as f:
        json.dump(json_features, f, indent=2, ensure_ascii=False)
    return feature_str

def feature_generation(llm: str, file_names_dict: dict, feature_str: str, img_ids: list | None = None) -> list[dict]:
    #Feature generation
    rows = []
    img_ids = img_ids or list(file_names_dict.keys())
    for img_id in tqdm(img_ids[:900]):
        image_file_path = "./data/zod_yolo/images/val/" + file_names_dict[img_id]
        img_query = generate_feature_query(feature_str, image_file_path, llm)
        
        img_feat_values_str = get_response(llm, img_query)
        img_feat_values_json = extract_json(img_feat_values_str)
        img_feat_dict = img_feat_values_json["features"]
        
        row = {dict['feature_name']: dict['answer'] for dict in img_feat_dict}
        row['img_id'] = img_id
        rows.append(row)
    return rows

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM feature extraction")
    parser.add_argument("llm", type=str, help="Select LLM provider, can be gemini or openai")
    parser.add_argument("--resume", action="store_true", help="Resume from existing features and descriptions")
    args = parser.parse_args()

    assert args.llm in ["gemini", "openai"]

    data = pd.read_csv("./data/metaf_yolo_500e.csv", index_col=0)
    img_file_names = os.listdir("./data/zod_yolo/images/val/")
    file_names_dict = get_files_by_ids(data.index.tolist(), img_file_names)
    dataset = {"df": data, 
                "prefix": "zod", 
                "name": "Object Detection Dataset", 
                "description": "High-quality, frame-centric autonomous driving dataset with dense 2D, designed to evaluate object detection systems.", 
                "image_path": "./data/zod_yolo/images/val/",
                "img_file_names": file_names_dict,
                #"target_column": "",
                }

    existing_df = None
    existing_feature_str = None
    if args.resume:
        if os.path.exists("./data/llm_metafeatures.csv"):
            existing_df = pd.read_csv("./data/llm_metafeatures.csv", index_col=0)
        if os.path.exists("./data/llm_metafeatures_description.json"):
            with open("./data/llm_metafeatures_description.json", "r") as f:
                existing_feature_str = json.dumps(json.load(f), indent=2, ensure_ascii=False)

    if existing_feature_str is None:
        feature_str = feature_discovery(args.llm, dataset)
    else:
        feature_str = existing_feature_str

    if existing_df is not None:
        missing_ids = [img_id for img_id in data.index.tolist() if img_id not in existing_df.index]
    else:
        missing_ids = data.index.tolist()

    if missing_ids:
        missing_files_dict = get_files_by_ids(missing_ids, img_file_names)
        img_ids_to_process = list(missing_files_dict.keys())
        rows = feature_generation(args.llm, missing_files_dict, feature_str, img_ids=img_ids_to_process)
        new_df = pd.DataFrame(rows).set_index("img_id")
        if existing_df is not None:
            df = pd.concat([existing_df, new_df])
        else:
            df = new_df
    else:
        df = existing_df if existing_df is not None else pd.DataFrame()

    print(df)
    df.to_csv("./data/llm_metafeatures.csv")
        
    

    
    

        
