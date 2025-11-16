from datasets import load_dataset
import pandas as pd

def prepare_dataset():
    dataset = load_dataset("ageron/tatoeba_mt_train", "eng-spa")

    df = pd.concat([
        dataset["validation"].to_pandas(), dataset["test"].to_pandas()
    ], axis=0)\
           .sample(frac=1, random_state=42)\
           .reset_index(drop=True)
    
    df[["source_text", "target_text"]].to_parquet("eng_spa.parquet")
    print("Data saved to eng_spa.parquet")
    
if __name__ == "__main__":
    prepare_dataset()