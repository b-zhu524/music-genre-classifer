from datasets import load_dataset
from transformers import pipeline
from IPython.display import Audio


def load_fleurs_dataset():
    fleurs = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(fleurs))

    classifier = pipeline("audio-classification", model="sanchit-gandhi/whisper-medium-fleurs-lang-id")

    print(classifier(sample["audio"]))


if __name__ == "__main__":
    load_fleurs_dataset()
