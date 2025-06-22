from datasets import load_dataset
from transformers import pipeline
from IPython.display import Audio
import gradio as gr



def load_gtzan_dataset():
    # load
    gtzan = load_dataset("marsyas/gtzan", "all")

    # create train-test split 
    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)

    # convert to genre
    id2label_fn = gtzan["train"].features["genre"].int2str 

    return gtzan, id2label_fn



def generate_audio(gtzan, id2label_fn):
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label_fn(example["genre"])


"""
Generates five audio samples from the GTZAN dataset and displays them in a Gradio interface
"""
def get_audio():
    with gr.Blocks() as demo:
        with gr.Column():
            for _ in range(4):
                audio, label = generate_audio(load_gtzan_dataset()[0], load_gtzan_dataset()[1])
                output = gr.Audio(audio, label=label)

    demo.launch(debug=True)


if __name__ == "__main__":
    get_audio()