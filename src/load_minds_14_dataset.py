from datasets import load_dataset
from transformers import pipeline


def load_minds_14_dataset():
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    classifier = pipeline("audio-classification",
                        model="anton-l/xtreme_s_xlsr_300m_minds14")

    print(classifier(minds[0]["audio"]))


if __name__ == "__main__":
    load_minds_14_dataset()

