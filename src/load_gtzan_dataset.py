from datasets import load_dataset

def load_gtzan_dataset():
    gtzan = load_dataset("ashraq/gtzan", "all")
    sample = next(iter(gtzan))
    print(sample)