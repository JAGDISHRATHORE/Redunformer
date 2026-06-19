from datasets import load_dataset

def load_evaluation_dataset(dataset_name: str = "Salesforce/wikitext", subset: str = "wikitext-2-raw-v1", split: str = "test"):
    print(f"Loading dataset {dataset_name} ({subset}), split: {split}...")
    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"

    dataset = load_dataset(dataset_name, subset, split=split)
    return dataset
