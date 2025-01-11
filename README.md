# general-classifier

**general-classifier** is a Python package designed for multi-topic text classification. It allows users to define multiple classification topics, manage categories within each topic, classify text data using various language models, evaluate classification performance, and iteratively improve classification prompts leveraging Large Language Models (LLMs).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Define Topics and Categories](#1-define-topics-and-categories)
  - [2. Set Models](#2-set-models)
  - [3. Classify a Dataset](#3-classify-a-dataset)
  - [4. Evaluate Prompt Performance](#4-evaluate-prompt-performance)
  - [5. Improve Prompts Iteratively](#5-improve-prompts-iteratively)
  - [6. Manage Topics and Categories](#6-manage-topics-and-categories)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Dynamic Topic & Category Management**:  
  Easily add, remove, and manage multiple classification topics and their respective categories.

- **Flexible Model Integration**:  
  Supports integration with various language models, including local Transformers models, OpenAI's API, and DeepInfra.

- **Text Classification**:  
  Classify input text across all defined topics, with support for single-topic classification.

- **Performance Evaluation**:  
  Evaluate classification accuracy against ground truth data stored in CSV files, generating detailed confusion matrices and accuracy metrics.

- **Iterative Prompt Improvement**:  
  Utilize LLM feedback to iteratively refine and improve classification prompts, enhancing model accuracy over time.

- **Data Persistence**:  
  Save and load topic configurations to/from JSON files for easy data management and portability.

## Installation

Ensure you have Python 3.7 or higher installed. Install the required dependencies using `pip`:

```bash
pip install torch openai guidance
```

*Note: You may need additional installations based on the models you intend to use (e.g., Transformers models).*

Clone the repository and install the package:

```bash
git clone https://github.com/general-classifier-py.git
cd general-classifier-py
pip install -e .
```

## Quick Start

### 1. Define Topics and Categories

Begin by defining classification topics and their respective categories.

```python
from general_classifier import add_topic, add_category, show_topics_and_categories

# Add a new topic with categories
topic_info = add_topic(
    topic_name="Car Brands",
    categories=["BMW", "Audi", "Mercedes"]
)

# Add another category to the existing topic
add_category(topicId=topic_info['id'], categoryName="Toyota")

# Display all defined topics and their categories
show_topics_and_categories()
```

### 2. Set Models

Configure the main classification model and the prompt improvement model.

```python
from general_classifier import setModel, setPromptModel

# Set the main classification model (e.g., a local Transformers model)
setModel(newModel="bert-base-uncased", newModelType="Transformers")

# Set the prompt improvement model (e.g., OpenAI's GPT-4)
setPromptModel(newPromptModel="gpt-4", newPromptModelType="OpenAI", api_key="your-openai-api-key")
```

### 3. Classify a Dataset

Classify text data from a CSV file and evaluate performance.

```python
from general_classifier import classify_table

# Classify data from 'data.csv' with evaluation enabled
classify_table(dataset="data", withEvaluation=True, constrainedOutput=True)
```

This will generate `data_(result).csv` containing classification results and performance metrics.

### 4. Evaluate Prompt Performance

Assess the performance of a specific topic's prompt on a dataset.

```python
from general_classifier import check_prompt_performance_for_topic

# Evaluate prompt accuracy for topic 'A' on dataset 'mydata.csv'
check_prompt_performance_for_topic(topicId="A", dataset="mydata", constrainedOutput=True)
```

### 5. Improve Prompts Iteratively

Enhance the classification prompt for a specific topic using LLM feedback.

```python
from general_classifier import improve_prompt

# Iteratively improve prompt for topic 'A' using dataset 'mydata.csv'
improve_prompt(topicId="A", dataset="mydata", constrainedOutput=True, num_iterations=10)
```

This function will refine the prompt over multiple iterations, seeking to improve classification accuracy.

### 6. Manage Topics and Categories

Additional functions to manage topics and categories:

- **Update a Topic's Prompt**:

  ```python
  from general_classifier import setPrompt

  # Update the prompt for topic 'A'
  setPrompt(topicId="A", newPrompt="New improved prompt for classification tasks.")
  ```

- **Remove a Specific Topic**:

  ```python
  from general_classifier import remove_topic

  # Remove topic with ID 'A'
  remove_topic("A")
  ```

- **Remove All Topics**:

  ```python
  from general_classifier import removeAllTopics

  # Remove all defined topics
  removeAllTopics()
  ```

## API Reference

### Functions

#### `setModel(newModel: str, newModelType: str, api_key: str = "")`

Sets the main classification model.

- **Parameters**:
  - `newModel`: The model identifier (e.g., `"bert-base-uncased"`, `"gpt-4"`).
  - `newModelType`: Type of the model (`"Transformers"`, `"OpenAI"`, `"DeepInfra"`).
  - `api_key`: API key for models that require authentication (default is empty).

#### `setPromptModel(newPromptModel: str, newPromptModelType: str, api_key: str = "")`

Sets the model used for prompt improvement.

- **Parameters**:
  - `newPromptModel`: The prompt model identifier.
  - `newPromptModelType`: Type of the prompt model (`"Transformers"`, `"OpenAI"`, `"DeepInfra"`).
  - `api_key`: API key for models that require authentication (default is empty).

#### `add_topic(topic_name: str, categories: list = [], condition: str = "", prompt: str = default_prompt) -> dict`

Adds a new classification topic.

- **Parameters**:
  - `topic_name`: Name of the topic.
  - `categories`: List of category names.
  - `condition`: Optional condition string.
  - `prompt`: Optional custom prompt.

- **Returns**:
  - `topic_info`: Dictionary containing topic details.

#### `remove_topic(topic_id_str: str)`

Removes a topic by its ID.

- **Parameters**:
  - `topic_id_str`: The ID of the topic to remove.

#### `add_category(topicId: str, categoryName: str, Condition: str = "")`

Adds a category to a specified topic.

- **Parameters**:
  - `topicId`: ID of the topic.
  - `categoryName`: Name of the new category.
  - `Condition`: Optional condition string.

#### `remove_category(topicId: str, categoryId: str)`

Removes a category from a specified topic.

- **Parameters**:
  - `topicId`: ID of the topic.
  - `categoryId`: ID of the category to remove.

#### `setPrompt(topicId: str, newPrompt: str)`

Updates the prompt for a specified topic.

- **Parameters**:
  - `topicId`: ID of the topic.
  - `newPrompt`: The new prompt string.

#### `removeAllTopics()`

Removes all defined topics and resets related counters and data structures.

#### `save_topics(filename: str)`

Saves all topics and their configurations to a JSON file.

- **Parameters**:
  - `filename`: Name of the JSON file to save topics.

#### `load_topics(filename: str)`

Loads topics and their configurations from a JSON file.

- **Parameters**:
  - `filename`: Name of the JSON file to load topics from.

#### `classify(text: str, isItASingleClassification: bool = True, constrainedOutput: bool = True, withEvaluation: bool = False, groundTruthRow: list = None) -> list`

Classifies a piece of text across all defined topics.

- **Parameters**:
  - `text`: The text to classify.
  - `isItASingleClassification`: If `True`, prints classification results.
  - `constrainedOutput`: If `True`, uses constrained output during classification.
  - `withEvaluation`: If `True`, evaluates against ground truth.
  - `groundTruthRow`: The ground truth row from the dataset for evaluation.

- **Returns**:
  - `ret`: List of predicted categories per topic.

#### `classify_table(dataset: str, withEvaluation: bool = False, constrainedOutput: bool = True)`

Classifies each row in a CSV dataset and evaluates performance.

- **Parameters**:
  - `dataset`: Base name of the CSV file (without `.csv`).
  - `withEvaluation`: If `True`, compares predictions to ground truth.
  - `constrainedOutput`: Controls output style in classification.

#### `check_prompt_performance_for_topic(topicId: str, dataset: str, constrainedOutput: bool = True, groundTruthCol: int = None)`

Evaluates the performance (accuracy) of a specific topic's prompt on a dataset.

- **Parameters**:
  - `topicId`: ID of the topic to evaluate.
  - `dataset`: Base name of the CSV file (without `.csv`).
  - `constrainedOutput`: Controls output style in classification.
  - `groundTruthCol`: Column index for ground truth; defaults to `(topic_index * 2) + 1`.

#### `improve_prompt(topicId: str, dataset: str, constrainedOutput: bool = True, groundTruthCol: int = None, num_iterations: int = 10)`

Iteratively improves the prompt for a specific topic using LLM feedback.

- **Parameters**:
  - `topicId`: ID of the topic to improve.
  - `dataset`: Base name of the CSV file (without `.csv`).
  - `constrainedOutput`: Controls output style in classification.
  - `groundTruthCol`: Column index for ground truth; defaults to `(topic_index * 2) + 1`.
  - `num_iterations`: Number of prompt improvement iterations.

### Classes

#### `MockText`

A simple mock class to mimic objects with a `.value` attribute, used for storing topic and category information.

```python
class MockText:
    def __init__(self, value: str):
        self.value = value
```

## Examples

### Example 1: Adding and Removing Topics

```python
from general_classifier import add_topic, remove_topic, show_topics_and_categories

# Add a new topic
topic_a = add_topic(
    topic_name="Fruit Types",
    categories=["Apple", "Banana", "Cherry"]
)

# Add another category to the existing topic
add_category(topicId=topic_a['id'], categoryName="Date")

# Display current topics
show_topics_and_categories()

# Remove a category
remove_category(topicId=topic_a['id'], categoryId='d')

# Remove the topic
remove_topic(topic_a['id'])

# Display topics after removal
show_topics_and_categories()
```

### Example 2: Classifying Text Data

```python
from general_classifier import setModel, classify_table

# Set the classification model
setModel(newModel="bert-base-uncased", newModelType="Transformers")

# Classify data without evaluation
classify_table(dataset="texts", withEvaluation=False, constrainedOutput=True)
```

### Example 3: Improving Prompts

```python
from general_classifier import setPromptModel, improve_prompt

# Set the prompt improvement model
setPromptModel(newPromptModel="gpt-4", newPromptModelType="OpenAI", api_key="your-openai-api-key")

# Improve prompt for a specific topic
improve_prompt(topicId="A", dataset="classification_data", constrainedOutput=True, num_iterations=5)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or new features.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


# Disclaimer

This tool is provided "as is" without any warranty. Use responsibly and ensure compliance with all relevant terms of service when integrating with external APIs like DeepInfra and OpenAI.

# Contact

For any questions or support, please open an issue on the [GitHub repository](https://github.com/general-classifier-py).

# Related Projects
The general classifier has been evaluated on oncological trials in the following projects: [GitHub repository](https://github.com/med-data-tools/general-classifier). 
The related publication has been published as a preprint on medrxiv - doi: https://doi.org/10.1101/2024.12.03.24318390

The general classifier is based on previous work using LLMs for title and abstract screening of biomedical publications
doi: https://doi.org/10.1186/s13643-024-02575-4; [GitHub repository](https://github.com/med-data-tools/title-abstract-screening-ai)