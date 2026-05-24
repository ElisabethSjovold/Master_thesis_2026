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
  
  """
  Generated model responses for all prompts in a DataFrame.
  
  The function supports two model types: 
  - "norwai": uses a prompt + "Svar" format
  - "normistral": uses the models chat template
  
  Parameters:
    df: DataFrame containing the prompts to generate responses for.
    model: The language model used for generation. 
    tokenizer: Tokenizer belonging to the model. 
    model_type: Either "norwAI" or "normistral"
    prompt_col: Name of the column containing the Norwegian prompts.
    target_col: Name of the column containing target responses, if available.
    system_prompt: Optional system prompt to prepend to each user prompt. If None, no system prompt is used.
    max_new_tokens: Maximum number of new tokens to generate.
    do_sample: Whether to use sampling during generation. 
    return_only_generated: If True, only the newly generated text is returned.
    
    Returns: 
        A list of dictionaries containing prompts, generated outputs and additional metadata."""

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


def save_generations(
        test_df,
        results,
        output_name,
        output_dir="Outputs",
        output_col="generated",
        result_key="model_output"
):
    """
    Saves the generated model outputs together with the original test DataFrame.
    
    Parameters: 
        test_df: Original DataFrame used for generation.
        results: List of generated outputs returned by generate_from_df.
        output_name: Name of the output file (without extension).
        output_dir: Directory to save the output file in.
        output_col: Name of the new column containing generated responses. 
        results_key: Key in results containing the generated model output. 
        
        Returns: 
        None. The function saves a JSONL file to disk."""
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


