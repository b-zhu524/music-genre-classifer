from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
from datasets import Audio
import load_gtzan_dataset
import numpy as np
import evaluate


def process_audio_files():
    model_id = "ntu-spml/distilhubert"  # distilled version of the HuBERT model 
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        do_normalize=True,  # normalize the audio signal
        return_attention_mask=True,
    )

    sampling_rate = feature_extractor.sampling_rate # 16000
    
    gtzan = load_gtzan_dataset.load_gtzan_dataset()[0]
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))  # resample audio to 16kHz

    sample = gtzan["train"][0]["audio"]
    # print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

    inputs = feature_extractor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
    )

    print(f"input keys: {list(inputs.keys())}")

    print(
        f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
    )   # input_values is the audio signal in the frequency domain; attention_mask is the mask for the input values

    return feature_extractor, gtzan, model_id 


def preprocess_function(examples):
    max_duration = 30.0

    audio_arrays = [x["array"] for x in examples["audio"]]

    feature_extractor = process_audio_files()[0]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length = int(feature_extractor.sampling_rate * max_duration),
        truncation = True,
        return_attention_mask = True
    )

    return inputs


def map_dataset():
    gtzan = process_audio_files()[1]
    gtzan_encoded = gtzan.map(
        preprocess_function,
        remove_columns = ["audio", "file"],
        batched = True,
        batch_size = 100,
        num_proc = 1
    )

    gtzan_encoded = gtzan_encoded.rename_column("genre", "label")

    id2label_fn = load_gtzan_dataset.load_gtzan_dataset()[1]
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(gtzan_encoded["train"].features["label"].names))
    }

    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)
    model_id = process_audio_files()[2]
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model_name = model_id.split("/")[-1]
    batch_size = 8
    gradient_accumulation_steps = 1
    num_train_epochs = 10


    training_args = TrainingArguments(
        f"{model_name}-finetuned-gtzan",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        push_to_hub=True,   # automatic upload of fine-tuned checkpoints during training
    )

    return gtzan_encoded, model, training_args


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    

def train_model():
    gtzan_encoded = map_dataset()[0]
    model = map_dataset()[1]
    training_args = map_dataset()[2]

    feature_extractor = process_audio_files()[0]
    trainer = Trainer(
        model,
        training_args,
        train_dataset=gtzan_encoded["train"],
        eval_dataset=gtzan_encoded["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print(trainer.train()) # training ~1HR


if __name__ == "__main__":
    map_dataset()
