import os
import re
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from pathlib import Path

class GPT2TextGenerator:
    def __init__(self, model_name='gpt2', device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device

    @staticmethod
    def read_txt(file_path):
        with open(file_path, "r") as file:
            text = file.read()
        return text

    @staticmethod
    def read_documents_from_directory():
        directory = os.path.join(Path(__file__).parent, 'data')
        combined_text = ""
        print(directory)
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if filename.endswith(".txt"):
                combined_text += GPT2TextGenerator.read_txt(file_path)
        return combined_text

    # def preprocess_data(dataset):
    #     # Modify this method to include start and end tokens in the sequences
    #     start_token = "<start>"
    #     end_token = " <end>"

    #     converted_dataset = ""

    #     def convert_qa_format(example):
    #         question = example['qText']
    #         answer = ', '.join(example['answers'])
    #         # return { "Question": question, 'Answer': answer + end_token }
    #         return f"[Q]: {question} \n[A]: {answer} {end_token} \n\n"

    #     for example in dataset:
    #         converted_dataset += convert_qa_format(example)

    #     return converted_dataset

    # data_path = 'trainmodel.json'
    # with open(data_path, 'r') as j:
    #     data = json.loads(j.read())
    # clean_data_text = preprocess_data(data)
    # with open('clean_data.txt', 'w') as f:

    @staticmethod
    def clean_text(text):
        return re.sub(r'\n+', '\n', text).strip()

    @staticmethod
    def load_dataset(file_path, tokenizer, block_size=128):
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

    @staticmethod
    def load_data_collator(tokenizer, mlm=False):
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

    def train(self, train_file_path, output_dir, overwrite_output_dir=True,
              per_device_train_batch_size=4, num_train_epochs=1000, save_steps=5000):

        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<end>']})
        train_dataset = self.load_dataset(train_file_path, tokenizer)
        data_collator = self.load_data_collator(tokenizer)

        model = GPT2LMHeadModel.from_pretrained(self.model_name).to(torch.device(self.device))
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)

    @staticmethod
    def load_model(model_path):
        return GPT2LMHeadModel.from_pretrained(model_path)

    @staticmethod
    def load_tokenizer(tokenizer_path):
        return GPT2Tokenizer.from_pretrained(tokenizer_path)

    def generate_text(self, model_path, sequence, max_length, stop_token="<end>"):
        model = self.load_model(model_path)
        tokenizer = self.load_tokenizer(model_path)
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.encode(stop_token)[0] if stop_token else None,
            pad_token_id=model.config.eos_token_id,
            temperature=0.1,
        )
        return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


if __name__ == "__main__":

    gpt_generator = GPT2TextGenerator()
    text_data = gpt_generator.read_documents_from_directory()
    cleaned_data = gpt_generator.clean_text(text_data)

    with open("data/final_train.txt", "w") as f:
        f.write(cleaned_data)

    gpt_generator.train(
        train_file_path="data/final_train.txt",
        output_dir="models",
        num_train_epochs=3,  # Reduced for demonstration
        save_steps=500
    )

    # generated_text = gpt_generator.generate_text(
    #     model_path="models",
    #     sequence="[Q] Why does Santa have three gardens?",
    #     max_length=50
    # )
    # print(generated_text)
