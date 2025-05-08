"""
classifier_updated.py

This module contains updated classes and functions for model inference, prompt improvement, topic management,
and CSV-based classification. The code was refactored to reduce global state by introducing the following classes:

• LLMManager – handles model setup, loading, unloading, and various inference methods.
• Topic – encapsulates a classification topic (with prompt and categories).
• TopicManager – handles topic creation, updating, persistence (save/load), and classification over CSV datasets.
• A minimal UI function (open_interface) demonstrates how one might connect IPython widgets.

Note: In your real project you could separate UI logic (which uses ipywidgets) from the core logic.
"""

import os
import csv
import json
import re
import time
import uuid
import gc
import ast

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from openai import OpenAI
from guidance import models, select, gen

# Set random seed for reproducibility
torch.manual_seed(0)


class StopOnTokens(StoppingCriteria):
    """Stops generation if the last token is among the specified stop tokens."""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1].item() in self.stop_token_ids:
            return True
        return False


class LLMManager:
    """
    Manager to set up and perform operations with a language model.
    
    This supports both local (Transformers) and cloud (OpenAI/DeepInfra) inference, 
    as well as an alternative approach using Guidance for prompt improvements.
    """
    def __init__(self):
        self.model_name = None
        self.model_type = "Transformers"       # "Transformers", "OpenAI", or "DeepInfra"
        self.inference_type = None             # "transformers", "guidance", or "cloud"
        self.api_key = None
        self.tokenizer = None
        self.model = None
        self.client = None
        self.guidance_model = None

        # For prompt-specific model (if separate)
        self.prompt_model_name = None
        self.prompt_model_type = None
        self.prompt_inference_type = None
        self.prompt_tokenizer = None
        self.prompt_model = None
        self.prompt_guidance_model = None

    def set_model(
        self, model_name: str, model_type: str = "Transformers",
        inference_type: str = "transformers", api_key: str = ""
    ):
        """Set up the main model based on the provided parameters."""
        self.model_name = model_name
        self.model_type = model_type
        self.inference_type = inference_type
        self.api_key = api_key

        if model_type == "Transformers":
            if inference_type == "transformers":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif inference_type == "guidance":
                self.guidance_model = models.Transformers(model_name, echo=False, trust_remote_code=True)
            else:
                raise ValueError("Invalid inference type for Transformers")
        elif model_type == "OpenAI":
            self.inference_type = "cloud"
            if api_key:
                self.client = OpenAI(api_key=api_key)
        elif model_type == "DeepInfra":
            self.inference_type = "cloud"
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        else:
            raise ValueError("Unsupported model type")

    def load_model(self):
        """Load (or reload) the model instance."""
        if not self.model_name:
            raise ValueError("Model name not set")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return self.model

    def unload_model(self):
        """Unload the model from GPU memory to free resources."""
        if self.model is not None:
            self.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.model
            gc.collect()
            self.model = None

    def set_prompt_model(
        self, prompt_model_name: str, prompt_model_type: str, 
        inference_type: str = "guidance", api_key: str = ""
    ):
        """Set up the prompt improvement model."""
        self.prompt_model_name = prompt_model_name
        self.prompt_model_type = prompt_model_type
        self.prompt_inference_type = inference_type

        if prompt_model_type == "Transformers":
            if inference_type == "transformers":
                self.prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_model_name, trust_remote_code=True)
                self.prompt_model = AutoModelForCausalLM.from_pretrained(
                    prompt_model_name,
                    device_map="cuda",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
                self.prompt_tokenizer.pad_token_id = self.prompt_tokenizer.eos_token_id
            elif inference_type == "guidance":
                # Reuse guidance model if same as main
                if prompt_model_name == self.model_name and self.guidance_model is not None:
                    self.prompt_guidance_model = self.guidance_model
                else:
                    self.prompt_guidance_model = models.Transformers(prompt_model_name, echo=False, trust_remote_code=True)
        elif prompt_model_type in ("OpenAI", "DeepInfra"):
            if api_key:
                base_url = "https://api.deepinfra.com/v1/openai" if prompt_model_type == "DeepInfra" else None
                self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("Unsupported prompt model type")

    def calculate_word_probability(self, prompt: str, target_word: str):
        """
        Calculate the probability of a target word following a given prompt.
        This method only applies when using a Transformers-based model.
        
        Returns:
            (total_probability, token_probabilities) tuple.
        """
        self.model.eval()
        if not prompt.endswith(" "):
            target_word = " " + target_word
        target_tokens = self.tokenizer.encode(target_word, add_special_tokens=False)
        token_probabilities = []
        current_text = prompt
        for token_id in target_tokens:
            inputs = self.tokenizer(current_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            token_prob = next_token_probs[token_id].item()
            token_probabilities.append(token_prob)
            # Append the predicted token and update current text
            encoded_text = self.tokenizer.encode(current_text, add_special_tokens=False)
            current_text = self.tokenizer.decode(encoded_text + [token_id], skip_special_tokens=True)
        total_probability = torch.tensor(token_probabilities).prod().item()
        return total_probability, token_probabilities

    def calculate_options_probabilities(self, prompt: str, options: list):
        """
        Calculate and compare the probabilities for different options following the given prompt.
        Returns:
            best_option, best_probability, all option probabilities (absolute and relative).
        """
        self.model.eval()
        space_prefix = " " if not prompt.endswith(" ") else ""
        first_token_groups = {}
        for option in options:
            first_token = self.tokenizer.encode(space_prefix + option, add_special_tokens=False)[0]
            first_token_groups.setdefault(first_token, []).append(option)
        base_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            base_outputs = self.model(**base_inputs)
            base_logits = base_outputs.logits[0, -1, :]
            base_probs = F.softmax(base_logits, dim=-1)
        option_probabilities = {}
        for token_id, group_options in first_token_groups.items():
            if len(group_options) == 1:
                option = group_options[0]
                token_ids = self.tokenizer.encode(space_prefix + option, add_special_tokens=False)
                if len(token_ids) == 1:
                    option_probabilities[option] = base_probs[token_id].item()
                else:
                    probability, _ = self.calculate_word_probability(prompt, option)
                    option_probabilities[option] = probability
            else:
                for option in group_options:
                    probability, _ = self.calculate_word_probability(prompt, option)
                    option_probabilities[option] = probability
        total_probability = sum(option_probabilities.values())
        relative_probabilities = {opt: (prob / total_probability if total_probability > 0 else 0)
                                  for opt, prob in option_probabilities.items()}
        best_option, best_prob = max(option_probabilities.items(), key=lambda x: x[1])
        return best_option, best_prob, option_probabilities, relative_probabilities, relative_probabilities[best_option]

    def get_answer(
        self, prompt: str, categories: list, constrained_output: bool = True,
        temperature: float = 0.0, think_step: int = 0
    ):
        """
        Get an answer using the specified inference method.
        For cloud-based inference, uses OpenAI client.
        For local inference it calls either 'guidance' or Transformers generation.
        
        Returns:
            (selected_category, relative_probability)
        """
        if self.inference_type == "cloud":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,
                temperature=temperature,
            )
            generated_answer = completion.choices[0].message.content
            for option in categories:
                if re.search(re.escape(option), generated_answer, re.IGNORECASE):
                    return option, "-"
            return "undefined", "-"
        elif self.inference_type == "guidance":
            if constrained_output:
                output = self.guidance_model + " " + prompt + select(options=categories, name="answer")
                answer = output["answer"]
            else:
                output = self.guidance_model + " " + prompt + gen(max_tokens=15, name="answer")
                generated_answer = output["answer"]
                answer = next((option for option in categories if re.search(re.escape(option), generated_answer, re.IGNORECASE)), "undefined")
            return answer, "-"
        elif self.inference_type == "transformers":
            # Optional extra reasoning ("think-step")
            if think_step > 0:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                period_id = self.tokenizer.encode(".", add_special_tokens=False)[-1]
                stopping_criteria = StoppingCriteriaList([StopOnTokens([period_id])])
                outputs = self.model.generate(
                    inputs.input_ids, temperature=0.01, max_new_tokens=100,
                    do_sample=True, top_k=50, top_p=0.95, stopping_criteria=stopping_criteria
                )
                prompt = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] + " Therefore the correct answer is '"
                print("New prompt after think-step:", prompt)
            if constrained_output:
                best_option, best_prob, _, _, _ = self.calculate_options_probabilities(prompt, categories)
                return best_option, best_prob
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    inputs.input_ids, max_length=100, num_return_sequences=1,
                    temperature=temperature, do_sample=True, top_k=50, top_p=0.95
                )
                generated_answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                answer = next((option for option in categories if re.search(re.escape(option), generated_answer, re.IGNORECASE)), "undefined")
                return answer, "-"
        else:
            raise ValueError("Unsupported inference type")

    def get_llm_improved_prompt_with_feedback(
        self, old_prompt: str, old_accuracy: float, topic_name: str, category_list: list
    ):
        """
        Asks the prompt-improvement LLM to provide an updated prompt.
        The improved prompt must include the placeholder [TEXT].
        """
        category_str = ", ".join(category_list) if category_list else "No categories defined"
        system_content = (
            f"You are an advanced prompt engineer.\n"
            f"The classification topic is '{topic_name}'.\n"
            f"The available categories for this topic are: {category_str}\n"
            "Rewrite the user's prompt to achieve higher accuracy on classification tasks.\n"
            "You MUST keep the placeholder [TEXT].\n"
            "IMPORTANT: Output ONLY the final prompt, wrapped in triple backticks.\n"
            "No commentary or explanations.\n"
            "The new prompt should be in English."
        )
        user_content = (
            f"Previously, the prompt achieved an accuracy of {old_accuracy:.2f}%.\n"
            "Here is the old prompt:\n\n"
            f"{old_prompt}\n\n"
            "Please rewrite/improve this prompt. Keep [TEXT]. Wrap your entire revised prompt in triple backticks."
        )
        if self.prompt_model_type in ("OpenAI", "DeepInfra"):
            try:
                completion = self.client.chat.completions.create(
                    model=self.prompt_model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                improved_prompt = completion.choices[0].message.content.strip()
                match = re.search(r"```(.*?)```", improved_prompt, flags=re.DOTALL)
                if match:
                    improved_prompt = match.group(1).strip()
                else:
                    print("Warning: No triple backticks found; using full text.")
                if not improved_prompt or "[TEXT]" not in improved_prompt:
                    print("Warning: Improved prompt is invalid; reverting to old prompt.")
                    return old_prompt
                return improved_prompt
            except Exception as e:
                print(f"Error calling cloud prompt improvement: {e}")
                return old_prompt
        else:
            try:
                base_instruction = system_content
                improvement_request = f"{base_instruction}\n\nOriginal prompt:\n{old_prompt}\n"
                if self.prompt_inference_type == "transformers":
                    inputs = self.prompt_tokenizer(improvement_request, return_tensors="pt").to("cuda")
                    period_id = self.prompt_tokenizer.encode(".", add_special_tokens=False)[-1]
                    stopping_criteria = StoppingCriteriaList([StopOnTokens([period_id])])
                    outputs = self.prompt_model.generate(
                        inputs.input_ids, temperature=0.01, max_new_tokens=100,
                        do_sample=True, top_k=50, top_p=0.95, stopping_criteria=stopping_criteria
                    )
                    new_prompt = self.prompt_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                elif self.prompt_inference_type == "guidance":
                    script = self.prompt_guidance_model + " " + improvement_request + gen(max_tokens=250, name="improvedPrompt")
                    new_prompt = script["improvedPrompt"]
                if not new_prompt or "[TEXT]" not in new_prompt:
                    print("Warning: Local improved prompt is invalid; reverting to old prompt.")
                    return old_prompt
                return new_prompt
            except Exception as e:
                print(f"Error in local prompt improvement: {e}")
                return old_prompt


# ------------------------------------------------------------------------------
# Topic and TopicManager
# ------------------------------------------------------------------------------

class Topic:
    """
    Encapsulates a single classification topic, its prompt, condition, and categories.
    """
    def __init__(self, topic_name: str, prompt: str = None, condition: str = "", categories: list = None):
        self.id = self.generate_id()
        self.topic_name = topic_name
        self.condition = condition
        # Default prompt if not provided
        self.prompt = prompt if prompt else (
            "INSTRUCTION: You are a helpful classifier. You select the correct category among the options for classifying a piece of text. "
            "The classification topic is '[TOPIC]'. The allowed categories are '[CATEGORIES]'. "
            "QUESTION: The text is '[TEXT]'. ANSWER: The correct category is '"
        )
        self.categories = []  # List of dicts with keys: id, value, condition
        if categories:
            for cat in categories:
                self.add_category(cat)
        self.best_prompt_found = None
        self.best_prompt_accuracy = None

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def number_to_letters(num: int, uppercase: bool = True) -> str:
        letters = ""
        while num > 0:
            num, remainder = divmod(num - 1, 26)
            letters = chr((65 if uppercase else 97) + remainder) + letters
        return letters

    def add_category(self, category_name: str):
        cat_id = self.number_to_letters(len(self.categories) + 1, uppercase=False)
        self.categories.append({"id": cat_id, "value": category_name, "condition": ""})

    def remove_category(self, cat_id: str):
        self.categories = [cat for cat in self.categories if cat["id"] != cat_id]

    def update_condition(self, condition: str):
        self.condition = condition

    def update_prompt(self, new_prompt: str):
        self.prompt = new_prompt

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "topic_input": self.topic_name,
            "condition": self.condition,
            "prompt": self.prompt,
            "categories": self.categories
        }


class TopicManager:
    """
    Manages a list of topics and provides functions for topic-related actions,
    including classification, CSV I/O, and evaluation.
    """
    def __init__(self):
        self.topics = []

    def add_topic(self, topic_name: str, categories: list = None, condition: str = "", prompt: str = None) -> Topic:
        topic = Topic(topic_name, prompt=prompt, condition=condition, categories=categories)
        self.topics.append(topic)
        return topic

    def remove_topic(self, topic_id: str):
        before = len(self.topics)
        self.topics = [t for t in self.topics if t.id != topic_id]
        if len(self.topics) < before:
            print(f"Topic {topic_id} removed.")
        else:
            print(f"No topic found with ID {topic_id}.")

    def add_category(self, topic_id: str, category_name: str, condition: str = ""):
        topic = self.get_topic(topic_id)
        if topic:
            topic.add_category(category_name)
            if condition:
                topic.update_condition(condition)
            print(f"Category '{category_name}' added to topic '{topic_id}'.")
        else:
            print(f"No topic found with ID {topic_id}.")

    def remove_category(self, topic_id: str, cat_id: str):
        topic = self.get_topic(topic_id)
        if topic:
            topic.remove_category(cat_id)
            print(f"Category {cat_id} removed from topic {topic_id}.")
        else:
            print(f"No topic found with ID {topic_id}.")

    def get_topic(self, topic_id: str) -> Topic:
        for t in self.topics:
            if t.id == topic_id:
                return t
        return None

    def save_topics(self, filename: str):
        data = [t.as_dict() for t in self.topics]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Topics saved to {filename}")

    def load_topics(self, filename: str):
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.topics = []
        for topic_data in data:
            topic = Topic(
                topic_data["topic_input"],
                prompt=topic_data.get("prompt"),
                condition=topic_data.get("condition")
            )
            for cat in topic_data.get("categories", []):
                topic.add_category(cat["value"])
            topic.id = topic_data.get("id", topic.id)
            self.topics.append(topic)
        print(f"Loaded {len(self.topics)} topic(s) from {filename}")

    def show_topics(self):
        if not self.topics:
            print("No topics defined.")
            return
        for i, topic in enumerate(self.topics, start=1):
            print(f"Topic {i} (ID={topic.id}): {topic.topic_name}")
            if topic.condition:
                print(f"  Condition: {topic.condition}")
            print(f"  Prompt: {topic.prompt}")
            if topic.categories:
                for j, cat in enumerate(topic.categories, start=1):
                    print(f"    {j}. {cat['value']} (ID={cat['id']})")
            else:
                print("    [No categories]")

    def classify_text(
        self, text: str, llm_manager: LLMManager, with_evaluation: bool = False,
        constrained_output: bool = True, ground_truth: dict = None
    ):
        """
        Classify a single text using every topic.
        ground_truth is an optional dict mapping topic IDs to true category names.
        
        Returns two dictionaries: results and probabilities (keyed by topic id).
        """
        results = {}
        probabilities = {}
        for topic in self.topics:
            # (For brevity, condition evaluation is not fully implemented here.)
            categories = [cat["value"] for cat in topic.categories]
            prompt = topic.prompt.replace("[TOPIC]", topic.topic_name)
            prompt = prompt.replace("[CATEGORIES]", str(categories))
            prompt = prompt.replace("[TEXT]", text)
            answer, rel_prob = llm_manager.get_answer(prompt, categories, constrained_output)
            results[topic.id] = answer
            probabilities[topic.id] = rel_prob
            if with_evaluation and ground_truth:
                print(f"Topic: {topic.topic_name}, Answer: {answer}, Ground Truth: {ground_truth.get(topic.id, 'N/A')}")
        return results, probabilities

    def classify_table(
        self, csv_filename: str, llm_manager: LLMManager,
        with_evaluation: bool = False, constrained_output: bool = True, batch_size: int = 100
    ):
        """
        Process a CSV file and write classification results along with (optional) evaluation metrics.
        The CSV is assumed to have text in its first column and ground-truth labels for each topic starting at column 2.
        """
        if not os.path.exists(csv_filename):
            print(f"CSV file {csv_filename} not found.")
            return
        output_filename = csv_filename.replace(".csv", "_results.csv")
        with open(csv_filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)
        result_header = ["Index", "Text"]
        for topic in self.topics:
            result_header.extend([f"{topic.topic_name} (GT)", f"{topic.topic_name} (Pred)", f"{topic.topic_name} (Prob)"])
        with open(output_filename, "w", newline="", encoding="utf-8") as wf:
            writer = csv.writer(wf, delimiter=";", quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(result_header)
            for i, row in enumerate(rows[1:], start=1):
                text = row[0]
                # Assume each topic's ground truth is in sequential columns (1-indexed after text)
                ground_truth = {}
                for j, topic in enumerate(self.topics, start=1):
                    if len(row) > j:
                        ground_truth[topic.id] = row[j].strip()
                preds, probs = self.classify_text(text, llm_manager, with_evaluation, constrained_output, ground_truth)
                result_row = [i, text]
                for topic in self.topics:
                    gt = ground_truth.get(topic.id, "")
                    pred = preds.get(topic.id, "")
                    prob = probs.get(topic.id, "")
                    result_row.extend([gt, pred, prob])
                writer.writerow(result_row)
        print(f"Classification complete. Results saved to {output_filename}")


# ------------------------------------------------------------------------------
# Minimal UI using ipywidgets
# ------------------------------------------------------------------------------

def open_interface(topic_manager: TopicManager, llm_manager: LLMManager):
    """
    Opens an IPython widgets–based interface to let a user input text for classification.
    This minimal example demonstrates the concept.
    """
    import ipywidgets as widgets
    from IPython.display import display

    text_input = widgets.Text(value="", description="Text to Classify:")
    classify_button = widgets.Button(description="Classify")
    output_area = widgets.Output()

    def on_classify_click(b):
        with output_area:
            output_area.clear_output()
            text = text_input.value
            results, _ = topic_manager.classify_text(text, llm_manager)
            for topic in topic_manager.topics:
                print(f"{topic.topic_name}: {results.get(topic.id)}")

    classify_button.on_click(on_classify_click)
    display(text_input, classify_button, output_area)


# ------------------------------------------------------------------------------
# Example usage (for testing purposes)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize LLM manager and set a model (example uses GPT-2; adjust as needed)
    llm_manager = LLMManager()
    llm_manager.set_model("gpt2", model_type="Transformers", inference_type="transformers")
    
    # Initialize Topic manager and create a sample topic
    topic_manager = TopicManager()
    sample_topic = topic_manager.add_topic("Sentiment Analysis", categories=["Positive", "Negative"])
    
    # Show current topics
    topic_manager.show_topics()
    
    # Classify a sample text
    sample_text = "I love this product!"
    results, probabilities = topic_manager.classify_text(sample_text, llm_manager)
    print("Classification Results:", results)
    
    # (Optional) To open the interactive UI in a Jupyter notebook, call:
    # open_interface(topic_manager, llm_manager)
