import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import re
from microbiome_encoder_v2 import MicrobiomeEncoderV2, ProjectionLayer

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
DATA_FILE = "/hd/liujx/microbiome_llm_project/data/microbiome_multitask_small.jsonl"
COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/multitask_v1"

NUM_SPECIES = 6374
EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

class MultiTaskLoss(nn.Module):
    def __init__(self, clm_weight=1.0, contrastive_weight=0.3, temperature=0.07):
        super().__init__()
        self.clm_weight = clm_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
    
    def forward(self, outputs, task_type, micro_embeds=None, text_embeds=None):
        loss = 0
        if outputs.loss is not None:
            loss += self.clm_weight * outputs.loss
        if task_type == "retrieval" and micro_embeds is not None and text_embeds is not None:
            contrastive_loss = self.info_nce_loss(micro_embeds, text_embeds)
            loss += self.contrastive_weight * contrastive_loss
        return loss
    
    def info_nce_loss(self, micro_embeds, text_embeds, temperature=0.07):
        micro_embeds = F.normalize(micro_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        sim_matrix = torch.matmul(micro_embeds, text_embeds.T) / temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = (loss_fct(sim_matrix, labels) + loss_fct(sim_matrix.T, labels)) / 2
        return loss

class MicrobiomeChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, counts_path):
        self.tokenizer = tokenizer
        self.samples = []
        print("正在加载 OTU 计数表...")
        self.otu_df = pd.read_csv(counts_path, sep='\t', index_col=0, skiprows=2, low_memory=False)
        if self.otu_df.shape[0] > self.otu_df.shape[1]:
            self.otu_df = self.otu_df.T
        self.otu_df = self.otu_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        print("正在加载训练样本...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"加载完成: {len(self.samples)} 个样本")
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item['messages']
        task_type = item.get('task_type', 'generation')
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        content = messages[0]['content']
        match = re.search(r"样本\s+([\d\.]+)", content)
        if match:
            sample_id = match.group(1)
        else:
            sample_id = content.split("样本 ")[1].split(" ")[0] if "样本 " in content else None
        if sample_id and str(sample_id) in self.otu_df.index.astype(str):
            otu_vector = self.otu_df.loc[sample_id].values.astype(np.float32)
        else:
            otu_vector = np.zeros(NUM_SPECIES, dtype=np.float32)
        inputs = self.tokenizer(text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "otu_vector": torch.tensor(otu_vector),
            "labels": inputs.input_ids.squeeze(),
            "task_type": task_type,
            "sample_id": str(sample_id)
        }

def main():
    print("🚀 正在初始化模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map={"": "cuda:1"}, trust_remote_code=True, local_files_only=True)
    llm = prepare_model_for_kbit_training(llm)
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    llm = get_peft_model(llm, lora_config)
    micro_encoder = MicrobiomeEncoderV2(num_species=NUM_SPECIES, hidden_size=EMBED_DIM, num_layers=2, dropout=0.1).to("cuda:1")
    projection = ProjectionLayer(input_dim=EMBED_DIM, output_dim=LLM_HIDDEN_SIZE).to("cuda:1")
    multi_task_loss = MultiTaskLoss(clm_weight=1.0, contrastive_weight=0.3, temperature=0.07).to("cuda:1")
    dataset = MicrobiomeChatDataset(DATA_FILE, tokenizer, COUNTS_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(list(llm.parameters()) + list(micro_encoder.parameters()) + list(projection.parameters()), lr=LEARNING_RATE)
    print("🔥 开始多任务指令微调...")
    global_step = 0
    for epoch in range(EPOCHS):
        llm.train()
        micro_encoder.train()
        projection.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        step_count = 0
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to("cuda:1")
            attention_mask = batch['attention_mask'].to("cuda:1")
            otu_vectors = batch['otu_vector'].to("cuda:1").float()
            labels = batch['labels'].to("cuda:1")
            task_types = batch['task_type']
            micro_embeds = micro_encoder(otu_vectors)
            micro_tokens = projection(micro_embeds).unsqueeze(1)
            inputs_embeds = llm.base_model.model.model.embed_tokens(input_ids)
            combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
            combined_mask = torch.cat([attention_mask, torch.ones_like(micro_tokens[..., 0])], dim=1)
            ignore_index = -100
            padded_labels = torch.cat([labels, torch.full((labels.shape[0], 1), ignore_index, device=labels.device)], dim=1)
            outputs = llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=padded_labels)
            text_embeds = None
            if task_types[0] == "retrieval":
                with torch.no_grad():
                    last_hidden = outputs.logits.mean(dim=1)
                    text_embeds = last_hidden
            loss = multi_task_loss(outputs, task_types[0], micro_embeds=micro_embeds, text_embeds=text_embeds)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 100 == 0:
                    save_dir = os.path.join(OUTPUT_DIR, f"step_{global_step}")
                    llm.save_pretrained(save_dir)
                    torch.save({'micro_encoder': micro_encoder.state_dict(), 'projection': projection.state_dict()}, os.path.join(save_dir, "custom_layers.pt"))
                    print(f"💾 Checkpoint saved at step {global_step}")
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
        avg_loss = total_loss / step_count
        print(f"\n✅ Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f}")
    final_dir = os.path.join(OUTPUT_DIR, "final")
    llm.save_pretrained(final_dir)
    torch.save({'micro_encoder': micro_encoder.state_dict(), 'projection': projection.state_dict()}, os.path.join(final_dir, "custom_layers.pt"))
    print("\n🎉 训练全部完成！")

if __name__ == "__main__":
    main()
