import torch
from tqdm import tqdm
from pathlib import Path


def generate_from_df(df,
                     model,
                     tokenizer,
                     model_type,
                     prompt_col="Norwegian prompt",
                     system_prompt = None,
                     max_new_tokens=150,
                     do_sample=False,
                     return_only_generated=True):
  results = []
  model.eval()

  if system_prompt:
      print("System prompt used")
  else: 
      print("No system prompt used.")#

  for user_prompt in tqdm(df[prompt_col], total=len(df)):

    if model_type == "norwai":
        if system_prompt:
            full_prompt = (
              system_prompt.strip()
            + "\n\n"
            + user_prompt.strip()
            + "\n\nSvar:\n")

        else: 
            full_prompt = (
              user_prompt.strip()
            + "\n\nSvar:\n")
        
    elif model_type == "normistral":
        messages = []
        if system_prompt: 
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role":"user",
            "content":user_prompt})

        full_prompt = tokenizer.apply_chat_template( 
            messages,
            tokenize = False, 
            add_generation_prompt = True
        )
    

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample = do_sample,
          pad_token_id=tokenizer.eos_token_id
      )

    if return_only_generated:
          prompt_len = inputs["input_ids"].shape[-1]
          gen_only_ids = output_ids[0][prompt_len:]
          output_text = tokenizer.decode(
              gen_only_ids,
              skip_special_tokens=True).strip()
    else:
          output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


    results.append({
        "prompt": user_prompt,
        "full_prompt":full_prompt,
        "model_output": output_text,
        "system_prompt_used": system_prompt is not None
    })

  return results


#https://thepythoncode.com/article/calculate-rouge-score-in-python#rouge-l


# Function to save the generated outputs from all the models
def save_generations(
        test_df,
        results,
        output_name,
        output_dir="Outputs",
        output_col="generated",
        result_key="model_output"
):
    generated_texts = [r[result_key] for r in results]

    out_df = test_df.copy()
    out_df[output_col] = generated_texts

    Path(output_dir).mkdir(exist_ok=True)

    output_path = f"{output_dir}/{output_name}.jsonl"
    out_df.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"Saved {output_path}")


