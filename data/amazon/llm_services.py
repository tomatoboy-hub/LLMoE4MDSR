import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, BitsAndBytesConfig
import numpy as np

class LocalSummarizer:
    def __init__(self, model_name:str, use4bit:bool=False):
        print(f"Initializing LocalSummarizer with model {model_name} and use4bit {use4bit}")

        model_kwargs = {"device_map":"auto"}
        quantization_config = None
        if use4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 2. chat_templateがなければ、モデル名に応じて自動設定する
        # if self.tokenizer.chat_template is None:
        #     print("  WARNING: No chat template found in tokenizer. Applying a model-specific template.")
        #     if "gemma" in self.model_name.lower():
        #         print("  Applied Gemma chat template.")
        #         # Gemmaの公式Jinjaテンプレート
        #         self.tokenizer.chat_template = "{% for message in messages %}{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
        #     # elif "mistral" in self.model_name.lower():
        #     #     # 他のモデルも必要に応じてここに追加
        #     #     self.tokenizer.chat_template = ...
        #     else:
        #         print("  Could not determine a chat template for this model. Text generation might fail.")
        
        

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config,device_map="auto")

        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
        print("Summarizer \n")

    def summarize(self, prompt_text: str, max_new_tokens: int=150):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes user preferences based on item lists."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(prompt, max_new_tokens = max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9)
        return outputs[0]["generated_text"][len(prompt):].strip()

        
    def summarize_batch(self, prompt_texts: list[str], max_new_tokens: int=150, batch_size: int = 8):
        # 各テキストをチャットテンプレートに変換
        prompts = []
        for text in prompt_texts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes user preferences based on item lists."},
                {"role": "user", "content": text},
            ]
            prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # pipelineにリストとbatch_sizeを渡す
        outputs = self.pipeline(
            prompts, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.6, 
            top_p=0.9,
            batch_size=batch_size  # バッチサイズを指定
        )

        # 結果から生成された部分だけを抽出してリストで返す
        results = []
        for i, output in enumerate(outputs):
            generated_text = output[0]["generated_text"]
            # プロンプト部分の長さを引いて、生成されたテキストのみを抽出
            result = generated_text[len(prompts[i]):].strip()
            results.append(result)
            
        return results

class LocalEmbedder:
    def __init__(self,model_name:str):
        print(f"Initializing LocalEmbedder with model {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print("Embedder \n")

    def get_embeddings(self, texts: list, batch_size: int = 32, prefix:str = ""):
        all_embeddings = []
        texts_with_prefix = [prefix + text for text in texts]

        with torch.no_grad():
            for i in range(0, len(texts_with_prefix),batch_size):
                batch = self.tokenizer(
                    texts_with_prefix[i:i + batch_size],
                    padding=True,truncation=True, return_tensors="pt", max_length=512,
                ).to(self.device)

                outputs = self.model(**batch)
                last_hidden = outputs.last_hidden_state
                attention_mask = batch["attention_mask"]
                masked_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(),0.0)
                sum_hidden = torch.sum(masked_hidden,dim=1)
                sum_mask = torch.sum(attention_mask,dim=1,keepdim=True)
                embeddings = sum_hidden / sum_mask

                normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(normalized_embeddings.cpu().numpy())

        return np.array(all_embeddings)