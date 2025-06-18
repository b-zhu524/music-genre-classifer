from datasets import load_dataset
from transformers import pipeline


def load_speech_commands_dataset():
    speech_commands = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)

    sample = next(iter(speech_commands))
    classifier = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")

    print(classifier(sample["audio"]).copy())


if __name__ == "__main__":
    load_speech_commands_dataset()
