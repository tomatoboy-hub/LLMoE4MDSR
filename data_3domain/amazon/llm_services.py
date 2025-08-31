import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel

class LocalSummarizer:
    def __init__(self, model_name:str, use4bit:bool=False):
        print(f"Initializing LocalSummarizer with model {model_name} and use4bit {use4bit}")

        model_kwargs = {"device_map":"auto"}
        if use4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokeninzer)
        print("Summarizer \n")

    def summarize(self, prompt_text: str, max_new_tokens: int=150):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes user preferences based on item lists."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(prompt, max_new_tokens = max_new_tokens, do_sample=True, temperature=0.6, top_p=0.9)
        return outputs[0]["generated_text"][len(prompt):].strip()

class LocalEmbedder:
    def __init__(self,model_name:str):
        print(f"Initializing LocalEmbedder with model {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print("Embedder \n")

    def get_embeddings(self, texts: list, batch_size: int = 32, prefix:str = ""):
        all_embeddings = []
        texts_with_prefix = [prefix + text for text in texts]

        with torch.no_grad():
            for i in range(0, len(texts_with_prefix),batch_size):
                batch = self.tokenizer(
                    texts_with_prefix[i:i + batch_size],
                    padding=True, trancation=True, return_tensors="pt", max_length=512,
                ).to(self.device)

                outputs = self.mdoel(**batch)
                last_hidden = outputs.last_hidden_state
                attention_mask = batch["attention_mask"]
                masked_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(),0.0)
                sum_hidden = torch.sum(masked_hidden,dim=1)
                sum_mask = torch.sum(attention_mask,dim=1,keepdim=True)
                embeddings = sum_hidden / sum_mask

                normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(normalized_embeddings.cpu().numpy())

        return np.array(all_embeddings)