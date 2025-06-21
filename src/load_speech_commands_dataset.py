from datasets import load_dataset
from transformers import pipeline
from IPython.display import Audio


def load_speech_commands_dataset():
    speech_commands = load_dataset("speech_commands", "v0.02", split="validation", streaming=True)

    sample = next(iter(speech_commands))
    classifier = pipeline("audio-classification",
                          model="MIT/ast-finetuned-speech-commands-v2")

    print(classifier(sample["audio"]).copy())
    return sample


def get_audio(sample):
    Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])


if __name__ == "__main__":
    sample = load_speech_commands_dataset()
