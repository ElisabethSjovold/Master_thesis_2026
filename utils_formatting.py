import pandas as pd
from datasets import Dataset 

"""
Formatting utilities for grid search and supervised fine-tuning. 
This module contains helper functions for: 
- Formatting prompts and responses into model-specific templates 
- Creating SFT training dataframes
- Converting dataframes into HuggingFace datasets 
"""

def format_norwai(row):
    """
    Format a single row into Norwai instruction format.

    Structure:
    <prompt> 
    
    Svar: 
    <target response> 

    Used for supervised fine-tuning where the model learns to generate the response after the "Svar:" prefix.
    """
    return (
        row["Norwegian prompt"].strip()
        + "\n\nSvar:\n"
        + row["Norwegian target response"].strip()
    )

def format_normistral(tokenizer):
  """
  Returns a formatter function that converts a dataframe row into Normistral chat template format using the provided tokenizer. 

  Implemented as a factory function because the tokenizer must be passed in and used inside df.apply().
  """
  def formatter(row):
    message = [
      {"role":"user", "content": row["Norwegian prompt"]},
      {"role":"assistant", "content": row["Norwegian target response"]}
  ]

    return tokenizer.apply_chat_template(
        message,
        tokenize=False
  )
  return formatter

def make_text_dataset(df):
  """Converts a dataframe with a 'text' column into a HuggingFace Dataset.
  Only the text column is kept since SFFTrainer expects a dataset containing training text sequences 
  
  Returns: 
  datasets.Dataset
  """
  
  return Dataset.from_pandas(df[["text"]], preserve_index=False)

def make_sft_dataframe(df, formatter):
  """
  Create a dataframe for supervised fine-tuning by applying a formating function to each row and storing the result in a new column called "text" 
  Parameters: 
  df: pandas.DataFrame
    Input dataframe containing prompts and responses 
  formatter: function 
    Function that formats a row into a training text string 
  
  Returns: 
  pandas.DataFrame with a 'text' column. 
  """
  df = df.copy()
  df["text"] = df.apply(formatter, axis=1)
  return df

