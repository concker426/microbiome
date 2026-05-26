import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
from tqdm import tqdm
import json
import re

# ================= 配置区域 =================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct" 
DATA_FILE = "/hd/liujx/microbiome_llm_project/data/microbiome_qa_enhanced.jsonl"
COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/ibd_natural_lang_v2"

NUM_SPECIES = 6374 
EMBED_DIM = 768    
LLM_HIDDEN_SIZE = 3584 

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

EPOCHS = 3
BATCH_SIZE = 1 
LEARNING_RATE = 2e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

class MicrobiomeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class MicrobiomeChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, counts_path):
        self.tokenizer = tokenizer
        self.samples = []
        print("正在加载 OTU 计数表...")
        self.otu_df = pd.read_csv(counts_path, sep='\t', index_col=0, skiprows=2, low_memory=False)
        if self.otu_df.shape[0] > self.otu_df.shape[1]:
            self.otu_df = self.otu_df.T
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item['messages']
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 【关键修复】使用正则表达式精确提取样本 ID
        content = messages[0]['content']
        match = re.search(r"样本\s+([\d\.]+)", content)
        if match:
            sample_id = match.group(1)
        else:
            # 如果正则没匹配到，尝试原来的逻辑但只取第一个空格前的部分
            sample_id = content.split("样本 ")[1].split(" ")[0]
            
        # 确保 sample_id 在 DataFrame 中存在
        if sample_id not in self.otu_df.index:
            # 尝试转换为 float 再匹配（处理 488 vs 488.0 的情况）
            try:
                sample_id_float = float(sample_id)
                if sample_id_float in self.otu_df.index:
                    sample_id = sample_id_float
            except:
                pass
                
        otu_vector = self.otu_df.loc[sample_id].values.astype(np.float32)
        
        inputs = self.tokenizer(text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "otu_vector": torch.tensor(otu_vector),
            "labels": inputs.input_ids.squeeze()
        }

def main():
    print("🚀 正在初始化模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        device_map={"": "cuda:1"}, 
        trust_remote_code=True, local_files_only=True
    )
    
    llm = prepare_model_for_kbit_training(llm)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )
    llm = get_peft_model(llm, lora_config)
    
    micro_encoder = MicrobiomeEncoder(NUM_SPECIES, EMBED_DIM).to("cuda:1")
    projection = nn.Linear(EMBED_DIM, LLM_HIDDEN_SIZE).to("cuda:1")
    
    dataset = MicrobiomeChatDataset(DATA_FILE, tokenizer, COUNTS_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(
        list(llm.parameters()) + list(micro_encoder.parameters()) + list(projection.parameters()), 
        lr=LEARNING_RATE
    )
    
    print("🔥 开始自然语言指令微调 (Enhanced)...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        llm.train(); micro_encoder.train(); projection.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        
        for batch in pbar:
            input_ids = batch['input_ids'].to("cuda:1")
            attention_mask = batch['attention_mask'].to("cuda:1")
            otu_vectors = batch['otu_vector'].to("cuda:1").float()
            labels = batch['labels'].to("cuda:1")
            
            micro_embeds = micro_encoder(otu_vectors) 
            micro_tokens = projection(micro_embeds).unsqueeze(1) 
            
            inputs_embeds = llm.base_model.model.model.embed_tokens(input_ids)
            
            combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
            combined_mask = torch.cat([attention_mask, torch.ones_like(micro_tokens[..., 0])], dim=1)
            
            ignore_index = -100
            padded_labels = torch.cat([labels, torch.full((labels.shape[0], 1), ignore_index, device=labels.device)], dim=1)
            
            outputs = llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=padded_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step % 50 == 0:
                save_dir = os.path.join(OUTPUT_DIR, f"step_{global_step}")
                llm.save_pretrained(save_dir)
                torch.save({'micro_encoder': micro_encoder.state_dict(), 'projection': projection.state_dict()}, 
                           os.path.join(save_dir, "custom_layers.pt"))

        print(f"Epoch {epoch+1} 完成 | 平均 Loss: {total_loss / len(dataloader):.4f}")

    final_dir = os.path.join(OUTPUT_DIR, "final")
    llm.save_pretrained(final_dir)
    torch.save({'micro_encoder': micro_encoder.state_dict(), 'projection': projection.state_dict()}, 
               os.path.join(final_dir, "custom_layers.pt"))
    print("✅ 训练全部完成！")

if __name__ == "__main__":
    main()
