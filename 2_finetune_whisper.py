#finetune whisper
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor,WhisperTokenizer,WhisperProcessor,WhisperForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer
import sys, os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def finetune(language):
    try:
        match language:
            case "chinese":
                dataset_path = "avintech/chinese_children_speech"
            case "malay":
                dataset_path = "avintech/malay_batch1"
            case "tamil":
                dataset_path = "avintech/tamil_children_speech"
                
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(dataset_path, split="train", use_auth_token=True)
        common_voice["test"] = load_dataset(dataset_path, split="test", use_auth_token=True)
        common_voice = common_voice.remove_columns(["file_names", "original_script", "fluency"])
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language=language, task="transcribe")
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language=language, task="transcribe")
        common_voice = common_voice.cast_column("audio_path", Audio(sampling_rate=16000))

        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio_path"]
            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            # encode target text to label ids
            batch["labels"] = tokenizer(batch["spoken_text"]).input_ids
            return batch

        common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        model.generation_config.language = language
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
            
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

        metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            wer = 100 * metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        training_args = Seq2SeqTrainingArguments(
            output_dir="./whisper-medium-"+language,  # change to a repo name of your choice
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=True,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        processor.save_pretrained(training_args.output_dir)
        trainer.train()
        kwargs = {
            "dataset_tags": dataset_path,
            "dataset": dataset_path,  # a 'pretty' name for the training dataset
            "model_name": "Whisper medium " + language + " - Avin Tech",  # a 'pretty' name for our model
            "finetuned_from": "openai/whisper-medium",
            "tasks": "automatic-speech-recognition",
        }
        trainer.push_to_hub(**kwargs)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == "__main__":
    #finetune("chinese")
    finetune("tamil")