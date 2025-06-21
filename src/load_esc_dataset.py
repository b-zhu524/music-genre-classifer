from datasets import load_dataset
from transformers import pipeline
from IPython.display import Audio
import sounddevice as sd


def load_esc_dataset():
    dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
    sample = next(iter(dataset))["audio"]["array"]

    candidate_labels = ["Sound of a dog", "Sound of a vacuum cleaner"]

    classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    print(classifier(sample, candidate_labels=candidate_labels))

    sd.play(sample, samplerate=16000)
    sd.wait()  # Wait until the sound has finished playing


if __name__ == "__main__":
    load_esc_dataset()
