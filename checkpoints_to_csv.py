
from testing_script import load_model, chat
import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    question_file: Optional[str] = field(default=None, metadata={"help": "the file path"})
    model_name: Optional[str] = field(default="meta-llama/Llama-2-13b-chat-hf", metadata={"help": "the model name"})


def get_questions(csv_file_path):
    try:
        # Open the CSV file in read mode
        with open(csv_file_path, 'r', newline='') as csv_file:
            # Create a CSV DictReader object
            csv_reader = csv.DictReader(csv_file)
            
            # Iterate through the rows of the CSV data
            questions = []
            for row in csv_reader:
                # You can access values by column name (header)
                questions.append(row['question'])
            return questions
    except FileNotFoundError:
        print(f"The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # parse filename
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    csv_file_path = script_args.question_file
        
    # get questions from csv
    questions = get_questions(csv_file_path)

    # create list of all models
    base_model = script_args.model_name
    base_dirs = ["sft", "dpo"]
    checkpoints = ["checkpoint-100", "checkpoint-200", "checkpoint-300", "checkpoint-400", "checkpoint-500"]
    
    all_models = [{'model_name':base_model, 'model_name_or_path':None}] # untrained
    for base_dir in base_dirs:
        for checkpoint in checkpoints:
            all_models.append({'model_name':base_model, 'model_name_or_path':f"{base_dir}/{checkpoint}"})

    # create new csv file
    model_and_answer_pairs = {}
    for model_settings in all_models:
        model_name = model_settings['model_name']
        model_name_or_path = model_settings['model_name_or_path']

        # load model
        try:
            model, tokenizer = load_model(model_name, model_name_or_path)
        except Exception as e:
            print(f"Skipping {model_name_or_path} due to error: {e}")
            continue

        # ask all questions
        answers = []
        for question in questions:
            answer = chat(question, model, tokenizer)
            answers.append(answer)

        # add to dictionary
        model_and_answer_pairs[model_name_or_path] = answers

    # write to csv. each row is a question, each column is a model
    os.makedirs('output_data', exist_ok=True)
    with open('output_data/answers.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        models = list(model_and_answer_pairs.keys())
        models_renamed = [m if m is not None else "untrained" for m in models]
        writer.writerow(['question'] + models_renamed)
        for i, question in enumerate(questions):
            row = [question]
            for model_name_or_path in model_and_answer_pairs.keys():
                row.append(model_and_answer_pairs[model_name_or_path][i])
            writer.writerow(row)

