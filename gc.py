import csv
import re
import torch
import json
import torch
import ast
import uuid
import json
import os
from openai import OpenAI
from guidance import models, select, gen
torch.manual_seed(0)


global model
global modelType
global promptModel
global promptModelType
global previous_results
global topics
global selectOptions
global topic_id_counter
global interface


model=""
modelType="Transformers"
promptModel=""
promptModelType="Transformers"


topics = []


interface = False
previous_results = {}
topic_id_counter = 0



def setModel(newModel,newModelType,api_key=""):
    model=newModel
    modelType=newModelType
    global ModelGuidance 
    global client
    if modelType=="Transformers":
        ModelGuidance = models.Transformers(model, device_map='cuda', torch_dtype=torch.bfloat16, echo=False, trust_remote_code=True)
    if modelType=="OpenAI":
        if not api_key=="":
            client = OpenAI(api_key=apiKeyOpenAI)
    if modelType=="DeepInfra":
        if not api_key=="":
            client = OpenAI(api_key=api_key,base_url="https://api.deepinfra.com/v1/openai")


def setPromptModel(newPromptModel, newPromptModelType, api_key=""):
    global promptModel
    global promptModelType
    promptModel=newPromptModel
    promptModelType=newPromptModelType
    global promptModelGuidance 
    global client
    if promptModelType=="Transformers":
        if promptModel==Model:
            promptModelGuidance=ModelGuidance
        else:
            promptModelGuidance = models.Transformers(model, device_map='cuda', torch_dtype=torch.bfloat16, echo=False, trust_remote_code=True)
    if modelType=="OpenAI" or promptModelType=="OpenAI":
        if not api_key=="":
            client = OpenAI(api_key=apiKeyOpenAI)
    if modelType=="DeepInfra" or promptModelType=="DeepInfra":
        if not api_key=="":
            client = OpenAI(api_key=api_key,base_url="https://api.deepinfra.com/v1/openai")


    
        
def getAnswer(prompt, topicIndex, constrainedOutput, selectOptions, temperature=0.0):
    if modelType=="OpenAI" or modelType=="DeepInfra":
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=30,
            temperature=temperature,
        )
        generated_answer = completion.choices[0].message.content

        # Print the extracted content (the assistant's answer)
        print(generated_answer)
        for option in ast.literal_eval(selectOptions[topicIndex]):
            escaped_option = re.escape(option)

            if re.search(escaped_option, generated_answer, re.IGNORECASE):
                ret = option  # Return the matching option
                break
        else:
            ret = "undefined"
        return ret
        
    else:
        if constrainedOutput==True:
            output=ModelGuidance+f' '+prompt+select(options=ast.literal_eval(selectOptions[topicIndex]),name='answer')
            ret=output["answer"]   
        else:
            output=ModelGuidance+f' '+prompt+gen(max_tokens=15,name='answer')
            generated_answer = output["answer"]
            print(generated_answer)
            for option in ast.literal_eval(selectOptions[topicIndex]):
                escaped_option = re.escape(option)

                if re.search(escaped_option, generated_answer, re.IGNORECASE):
                    ret = option  # Return the matching option
                    break
            else:
                ret = "undefined"


        return ret
    
    
def getAnswerSingleTopic(prompt, categories, constrainedOutput):
    """
    A specialized classification function for a single topic, avoiding selectOptions.
    'categories' is a list of strings (e.g., ['red', 'blue', 'green']).
    """
    if modelType in ("OpenAI", "DeepInfra"):
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0,
        )
        generated_answer = completion.choices[0].message.content
        print(generated_answer)  # Debug output

        # Search for any of the categories in the generated_answer
        for option in categories:
            escaped_option = re.escape(option)
            if re.search(escaped_option, generated_answer, re.IGNORECASE):
                return option
        return "undefined"

    else:
        # Using your local Guidance-based model or another approach
        if constrainedOutput:
            # Build a Guidance prompt that includes only these categories
            category_str = "[" + ",".join(f"'{cat}'" for cat in categories) + "]"
            output = ModelGuidance + f" {prompt}" + select(options=ast.literal_eval(category_str), name='answer')
            return output["answer"]
        else:
            output = ModelGuidance + f" {prompt}" + gen(max_tokens=15, name='answer')
            generated_answer = output["answer"]
            print(generated_answer)
            for option in categories:
                escaped_option = re.escape(option)
                if re.search(escaped_option, generated_answer, re.IGNORECASE):
                    return option
            return "undefined"

def evaluate_condition(condition):
    """
    Evaluates the condition based on previous classification results.
    Returns True if the condition is met, or True if empty.
    """
    if not condition:  # Empty condition is always true
        return True

    if "==" not in condition:
        print(f"Invalid condition format: {condition}")
        return False

    left_side, right_side = condition.split("==", 1)
    left_side = left_side.strip()
    right_side = right_side.strip()

    # The condition format is: TopicID == CategoryID
    # Check if we have a previous classification result for the topic
    if left_side not in previous_results:
        print(f"No previous classification result for topic '{left_side}'. Condition: {condition}")
        return False

    chosen_cat_id = previous_results[left_side]

    # Compare the previously chosen category ID to the right side of the condition
    return chosen_cat_id == right_side


def singleClassification(text, isItASingleClassification=True, constrainedOutput=True, withEvaluation=False, groundTruthRow=None):

    selectOptions = []
    for topic_data in topics:
        # build each topic's category list
        tmpSelectOptions = "["
        for category_input, _, _ in topic_data['categories']:
            tmpSelectOptions += "'" + category_input.value + "',"
        tmpSelectOptions = tmpSelectOptions[:-1] + "]"
        selectOptions.append(tmpSelectOptions)

    ret = []

    # If we are evaluating, conditions depend on groundTruth.
    # So we fill previous_results based on groundTruth before checking conditions.
    if withEvaluation and groundTruthRow is not None:
        for i, topic_info in enumerate(topics):
            groundTruthCategoryName = groundTruthRow[i+1]  # Assuming the first column in row is text and ground truths start at index 1
            # Find the category_id that matches the groundTruthCategoryName
            gt_cat_id = None
            for (cat_input, _, cat_id) in topic_info['categories']:
                if cat_input.value == groundTruthCategoryName:
                    gt_cat_id = cat_id
                    break
            previous_results[topic_info['id']] = gt_cat_id

    for l in range(len(selectOptions)):
        condition = topics[l]['condition'].value.strip()
        condition_is_true = evaluate_condition(condition)

        if not condition_is_true:
            # If condition is not met, skip classification
            ret.append("")
            if interface == True and isItASingleClassification:
                print(f"Skipping {topics[l]['topic_input'].value} due to unmet condition: {condition}")
            continue

        # Condition met or no condition:
        prompt = topics[l]['prompt'].value
        prompt = prompt.replace('[TOPIC]', topics[l]['topic_input'].value)
        prompt = prompt.replace('[CATEGORIES]', selectOptions[l])
        prompt = prompt.replace('[TEXT]', text)

        answer = getAnswer(prompt, l, constrainedOutput, selectOptions, 0.0)
        ret.append(answer)

        if not withEvaluation:
            # If we are not evaluating with groundTruth, store the predicted category for future conditions
            chosen_category_id = None
            for category_input, _, category_id in topics[l]['categories']:
                if category_input.value == answer:
                    chosen_category_id = category_id
                    break
            previous_results[topics[l]['id']] = chosen_category_id

        if interface == True and isItASingleClassification:
            print(f"{topics[l]['topic_input'].value}: {answer}")

    return ret

        
def get_current_accuracy(topic_info):
    """
    Reads topic_info['performance_label'].value which might look like "Accuracy: 82.34%".
    Parses out the numeric value, returning a float. If not found, return 0.0.
    """
    label_text = topic_info['performance_label'].value
    match = re.match(r"Accuracy:\s+([\d.]+)%", label_text)
    if match:
        return float(match.group(1))
    return 0.0



                

def generate_id():
    return str(uuid.uuid4())[:8] 

def number_to_letters(num, uppercase=True):
    """
    Converts a number to a letter-based ID.
    For example:
      1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, 28 -> AB, ...
    """
    letters = ""
    while num > 0:
        num -= 1
        letters = chr((num % 26) + (65 if uppercase else 97)) + letters
        num //= 26
    return letters

        


        


            
# Optional: Code to process all topics and categories on another button click
def show_topics_and_categories():
    """
    Print out all topics (and their categories) in 'topics'.
    Also prints out the topic prompt and condition if available.
    """
    if not topics:
        print("No topics are currently defined.")
        return

    for i, topic_info in enumerate(topics, start=1):
        # Basic topic info
        topic_name = topic_info['topic_input'].value
        topic_id = topic_info.get('id', '?')
        
        # Optional fields
        condition_val = topic_info['condition'].value if 'condition' in topic_info else None
        prompt_val    = topic_info['prompt'].value    if 'prompt'    in topic_info else None
        
        print(f"Topic {i} (ID={topic_id}): {topic_name}")

        # Print condition if present
        if condition_val:
            print(f"  Condition: {condition_val}")

        # Print prompt if present
        if prompt_val:
            print(f"  Prompt: {prompt_val}")

        # Print categories
        categories = topic_info.get('categories', [])
        if not categories:
            print("    [No categories in this topic]")
        else:
            for j, (category_input, _, cat_id) in enumerate(categories, start=1):
                cat_name = category_input.value
                print(f"    {j}. {cat_name} (ID={cat_id})")

    print()  # optional extra newline

   
def add_topic(topic_name, 
              categories=[], 
              condition="", 
              prompt="INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
        "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
        "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
        "ANSWER: The correct category for this text is '"):
   
    global topic_id_counter
    topic_id_counter += 1
    
    # We can reuse the same default prompt from your UI code, if no prompt is given
    if prompt is None:
        prompt = (
            "INSTRUCTION: You are a helpful classifier. You select the correct of the possible categories "
            "for classifying a piece of text. The topic of the classification is '[TOPIC]'. "
            "The allowed categories are '[CATEGORIES]'. QUESTION: The text to be classified is '[TEXT]'. "
            "ANSWER: The correct category for this text is '"
        )

    # Convert topic_name, condition, prompt into our minimal .value structure
    topic_input_mock = MockText(topic_name)
    condition_mock = MockText(condition)
    prompt_mock = MockText(prompt)
    
    # Create a topic ID label like A, B, C, ...
    topic_id = number_to_letters(topic_id_counter, uppercase=True)
    
    # Build the new topic_info structure
    topic_info = {
        'id': topic_id,
        'topic_input': topic_input_mock,
        'condition': condition_mock,
        'categories': [],
        'prompt': prompt_mock,
        # The UI code also references these, but we'll set them to None
        # or omit them if you prefer:
        'categories_container': None,
        'topic_box': None,
        'performance_label': None,
        'checkPrompt_button': None,
        'num_iterations_input': None,
        'iteratePromptImprovements_button': None,
        'replacePrompt_button': None,
        
        'best_prompt_found': None,
        'best_prompt_accuracy': None,
        
        # We'll track how many categories we've added:
        'category_counter': 0
    }
    
    # Now add each category in categories
    for cat_str in categories:
        topic_info['category_counter'] += 1
        cat_id = number_to_letters(topic_info['category_counter'], uppercase=False)  # a, b, c ...
        
        # We mimic the UI structure: (category_input, category_box, cat_id)
        # - category_input => MockText(cat_str)
        # - category_box => None (since no UI)
        # - cat_id => e.g. 'a'
        category_tuple = (MockText(cat_str), None, cat_id)
        
        topic_info['categories'].append(category_tuple)
    
    # Finally, store it in the global topics list
    topics.append(topic_info)
    
    return topic_info


def remove_topic(topic_id_str):
    """
    Removes the topic with the given ID from the global 'topics' list, if present.
    If no topic matches the ID, prints a message and does nothing.
    """
    for i, t in enumerate(topics):
        # Safely get 'id' from the topic (fall back to None if missing)
        if t.get('id') == topic_id_str:
            del topics[i]
            print(f"Topic (ID={topic_id_str}) removed.")
            return  # stop after removing the first match

    print(f"No topic found with ID={topic_id_str}.")
    
    
    
def add_category(topicId, categoryName, Condition=""):
    """
    Adds a new category to the topic with the given ID. If 'Condition' is not empty,
    this also updates the topic's condition.

    :param topicId: ID of the topic (e.g. 'A' or 'B') to which we add a category
    :param categoryName: The name of the new category (e.g. "BMW")
    :param Condition: (optional) If not empty, will set/overwrite the topic's condition
    """
    # 1) Find the matching topic by ID
    found_topic = None
    for topic_info in topics:
        if topic_info.get('id') == topicId:
            found_topic = topic_info
            break

    if not found_topic:
        print(f"No topic found with ID={topicId}")
        return

    # 2) Ensure 'category_counter' is defined
    if 'category_counter' not in found_topic:
        found_topic['category_counter'] = 0

    # 3) Create the category tuple
    found_topic['category_counter'] += 1
    cat_id = number_to_letters(found_topic['category_counter'], uppercase=False)
    new_category_tuple = (MockText(categoryName), None, cat_id)

    # 4) Append to the topic's 'categories' list
    if 'categories' not in found_topic:
        found_topic['categories'] = []
    found_topic['categories'].append(new_category_tuple)

    # 5) If Condition is provided, update the topic's condition
    if Condition:
        # Make sure the topic has a 'condition' field with a .value
        if 'condition' not in found_topic or not hasattr(found_topic['condition'], 'value'):
            found_topic['condition'] = MockText("")
        found_topic['condition'].value = Condition

    print(f"Category '{categoryName}' (ID={cat_id}) added to topic '{topicId}'.")
    if Condition:
        print(f"  Updated topic condition to: {Condition}")
        
        
def remove_category(topicId, categoryId):
    """
    Removes the category with 'categoryId' from the topic with 'topicId'.
    If the topic or category does not exist, it prints an error message.
    """
    # 1) Find the matching topic in the global 'topics' list
    for topic_info in topics:
        if topic_info.get('id') == topicId:
            # 2) Look through the topic's 'categories' list for the matching categoryId
            categories = topic_info.get('categories', [])
            for i, (cat_input, cat_box, cat_id) in enumerate(categories):
                if cat_id == categoryId:
                    del categories[i]
                    print(f"Removed category (ID={categoryId}) from topic (ID={topicId}).")
                    return
            
            # If we reach here, no category matched categoryId
            print(f"Category with ID='{categoryId}' not found in topic (ID={topicId}).")
            return

    # If we reach here, no topic matched topicId
    print(f"No topic found with ID='{topicId}'.")
    
    
    
def save_topics(filename):
    data = []
    for topic_info in topics:
        # Build a dict for JSON serialization
        topic_data = {
            'id': topic_info.get('id', ''),
            'topic_input': topic_info['topic_input'].value if 'topic_input' in topic_info else '',
            'condition': topic_info['condition'].value if 'condition' in topic_info else '',
            'prompt': topic_info['prompt'].value if 'prompt' in topic_info else '',
            'categories': []
        }

        # Extract category info
        for (cat_input, _, cat_id) in topic_info.get('categories', []):
            # cat_input is a mock or widget with .value
            cat_name = cat_input.value
            topic_data['categories'].append({
                'id': cat_id,
                'value': cat_name
            })

        data.append(topic_data)

    # Write the list of topic dicts to a JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Topics saved to {filename}")
    
def load_topics(filename):
    global topics
    topics.clear()  # Remove any existing topics

    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data should be a list of topic dicts
    for topic_data in data:
        # Create a new topic dictionary, mimicking your code structure
        new_topic = {
            'id': topic_data.get('id', ''),
            'topic_input': MockText(topic_data.get('topic_input', '')),
            'condition': MockText(topic_data.get('condition', '')),
            'prompt': MockText(topic_data.get('prompt', '')),
            'categories': [],
            'category_counter': 0
        }

        # Rebuild the categories list as (MockText(...), None, cat_id)
        for cat_dict in topic_data.get('categories', []):
            cat_id = cat_dict.get('id', '')
            cat_value = cat_dict.get('value', '')
            new_topic['category_counter'] += 1
            new_topic['categories'].append(
                (MockText(cat_value), None, cat_id)
            )

        topics.append(new_topic)

    print(f"Loaded {len(topics)} topic(s) from {filename}")
    
    
def add_condition(topicId, categoryId, conditionStr):
    """
    Adds (or overwrites) a condition to the category with 'categoryId'
    in the topic with 'topicId'.
    
    Internally, this changes the category from a 3-tuple:
        (cat_input, cat_box, cat_id)
    into a 4-tuple:
        (cat_input, cat_box, cat_id, cat_condition).
    """
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])
    
    for i, cat_tuple in enumerate(categories):
        # cat_tuple might be 3-tuple or 4-tuple
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = ""  # no condition yet
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            # Create a new 4-tuple with the updated condition
            new_cat_tuple = (cat_input, cat_box, cat_id, conditionStr)
            categories[i] = new_cat_tuple
            print(f"Condition '{conditionStr}' added to category (ID={categoryId}) in topic (ID={topicId}).")
            return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")
    
    
def remove_condition(topicId, categoryId):
    """
    Removes (clears) the condition for the category with 'categoryId'
    in the topic with 'topicId'.
    
    If that category is stored as a 4-tuple, sets the 4th element to "".
    If it's only a 3-tuple, there's nothing to remove.
    """
    found_topic = None
    for topic in topics:
        if topic.get('id') == topicId:
            found_topic = topic
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    categories = found_topic.get('categories', [])

    for i, cat_tuple in enumerate(categories):
        if len(cat_tuple) == 3:
            (cat_input, cat_box, cat_id) = cat_tuple
            cat_condition = None  # it never had a condition
        else:
            (cat_input, cat_box, cat_id, cat_condition) = cat_tuple

        if cat_id == categoryId:
            # If it was already a 3-tuple, no condition to remove
            if len(cat_tuple) == 3:
                print(f"Category (ID={categoryId}) in topic (ID={topicId}) has no condition.")
                return
            else:
                # Overwrite the 4th element with an empty string
                new_cat_tuple = (cat_input, cat_box, cat_id, "")
                categories[i] = new_cat_tuple
                print(f"Condition removed from category (ID={categoryId}) in topic (ID={topicId}).")
                return

    print(f"No category (ID={categoryId}) found in topic (ID={topicId}).")

    
    
def classify_table(dataset, withEvaluation=False, constrainedOutput=True):
    """
    Classifies each row in 'dataset.csv' (semicolon-separated) using the global 'topics' list.
    If withEvaluation=True, it compares predicted categories to the 'groundTruth' columns
    (assuming each topic's groundTruth is in the subsequent columns).

    Results are saved to:
      - dataset_(result).csv  (includes row-by-row classification)
    If withEvaluation=True, also appends metrics (TP, FP, FN, TN, accuracy, etc.).

    :param dataset: The base name of the CSV file (without ".csv").
    :param withEvaluation: Whether to compare predictions to ground truth in subsequent columns.
    :param constrainedOutput: Passes to singleClassification() to control output style.
    """
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return

    # We assume 'topics' and 'singleClassification' are defined elsewhere as globals.
    # For each topic, we build a confusion matrix of {cat_name -> {TP, FP, FN, TN}}
    categoryConfusions = []
    for i, topic_info in enumerate(topics):
        cat_map = {}
        for (cat_input, _, _cat_id) in topic_info['categories']:
            cat_name = cat_input.value
            cat_map[cat_name] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        categoryConfusions.append(cat_map)

    # We'll track how many correct predictions per topic, how many attempts per topic
    numberOfCorrectResults = []
    numberOfRelevantAttempts = []

    # We'll also track row-by-row classification results to write them out
    resultRows = []

    # By default, the code snippet used these variables:
    startcount = 1
    endcount = -1
    saveName = dataset + "_(result)"

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        readerlist = list(reader)

        # We iterate through the CSV rows
        count = -1
        for row in readerlist:
            count += 1
            if endcount != -1 and count > endcount:
                break

            if count == 0:
                # First row is assumed a header row:
                # We create a header for the output "single" CSV
                singleResult = [""]
                elementcounter = -1
                for element in row:
                    elementcounter += 1
                    if elementcounter == 0:
                        singleResult.append(element)  # e.g., "text"
                    else:
                        # For each topic/column beyond the first:
                        numberOfCorrectResults.append(0)
                        numberOfRelevantAttempts.append(0)
                        if withEvaluation:
                            singleResult.append(element + "(GroundTruth)")
                        singleResult.append(element)  # Another column to hold predictions
                # Write header to file
                with open(saveName + ".csv", 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(singleResult)

            else:
                # Actual data row
                if count >= startcount:
                    # We classify row[0] (the text) with singleClassification
                    if withEvaluation:
                        # Provide the ground truth row so singleClassification can store it
                        result = singleClassification(
                            row[0],
                            isItASingleClassification=False,
                            constrainedOutput=constrainedOutput,
                            withEvaluation=True,
                            groundTruthRow=row
                        )
                    else:
                        result = singleClassification(
                            row[0],
                            isItASingleClassification=False,
                            constrainedOutput=constrainedOutput
                        )

                    # 'result' is a list of predicted categories (one per topic).
                    # We'll update confusion matrix if we have groundTruth
                    for tIndex, predCategory in enumerate(result):
                        groundTruth = ""
                        if withEvaluation and (tIndex + 1) < len(row):
                            groundTruth = row[tIndex + 1].strip()

                        if groundTruth:
                            # For each possible category in this topic
                            for cat_name, conf_map in categoryConfusions[tIndex].items():
                                if cat_name == groundTruth and cat_name == predCategory:
                                    conf_map["TP"] += 1
                                elif cat_name != groundTruth and cat_name == predCategory:
                                    conf_map["FP"] += 1
                                elif cat_name == groundTruth and cat_name != predCategory:
                                    conf_map["FN"] += 1
                                else:
                                    # cat_name != groundTruth and cat_name != predCategory
                                    conf_map["TN"] += 1

                    # Build row for output CSV
                    singleResult = [str(count), row[0]]  # row ID & text
                    tmpCount = 0
                    for ret in result:
                        tmpCount += 1
                        if withEvaluation and tmpCount < len(row):
                            # groundTruth for this topic
                            ground_truth = row[tmpCount].strip()
                            if ground_truth:
                                # We have a relevant attempt
                                numberOfRelevantAttempts[tmpCount - 1] += 1
                                singleResult.append(ground_truth)  # the ground truth
                                singleResult.append(ret)           # the prediction

                                # If we predicted correctly
                                if ret == ground_truth:
                                    numberOfCorrectResults[tmpCount - 1] += 1
                            else:
                                # groundTruth is empty => skip metric updates
                                singleResult.append("")
                                singleResult.append(ret)
                        else:
                            # No groundTruth or no evaluation
                            if not withEvaluation:
                                singleResult.append(ret)
                            else:
                                singleResult.append("UNDEFINED")

                    # Append row to file
                    with open(saveName + ".csv", 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
                        writer.writerow(singleResult)

                    # Optionally store in memory if needed
                    resultRows.append(singleResult)

    # If we have groundTruth (withEvaluation=True), compute summary metrics
    if withEvaluation:
        with open(saveName + ".csv", 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

            # Header for the summary rows
            writer.writerow([
                "Topic", "Accuracy", "Correct Attempts", "Relevant Attempts",
                "Micro Acc", "Micro Prec", "Micro Recall", "Micro F1",
                "TP", "FP", "FN", "TN"
            ])

            # For each topic, sum the confusion matrix & compute metrics
            for i, topic_info in enumerate(topics):
                sumTP = 0
                sumFP = 0
                sumFN = 0
                sumTN = 0

                cat_map = categoryConfusions[i]
                for cat_name, conf_map in cat_map.items():
                    sumTP += conf_map["TP"]
                    sumFP += conf_map["FP"]
                    sumFN += conf_map["FN"]
                    sumTN += conf_map["TN"]

                # Accuracy per topic (simple ratio of correct attempts to relevant attempts)
                if numberOfRelevantAttempts[i] > 0:
                    accuracy = (numberOfCorrectResults[i] / numberOfRelevantAttempts[i]) * 100.0
                else:
                    accuracy = -1

                # Micro metrics
                micro_accuracy = (sumTP / (sumTP + sumFN)) if (sumTP + sumFN) > 0 else 0.0
                micro_precision = (sumTP / (sumTP + sumFP)) if (sumTP + sumFP) > 0 else 0.0
                micro_recall = (sumTP / (sumTP + sumFN)) if (sumTP + sumFN) > 0 else 0.0
                micro_f1 = 0.0
                if micro_precision > 0 and micro_recall > 0:
                    micro_f1 = 2.0 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

                topic_name = topic_info['topic_input'].value
                writer.writerow([
                    topic_name,
                    f"{accuracy:.2f}%",
                    numberOfCorrectResults[i],
                    numberOfRelevantAttempts[i],
                    f"{micro_accuracy*100:.2f}%",
                    f"{micro_precision*100:.2f}%",
                    f"{micro_recall*100:.2f}%",
                    f"{micro_f1*100:.2f}%",
                    sumTP,
                    sumFP,
                    sumFN,
                    sumTN
                ])

    # Optionally print final status
    print(f"Classification of '{dataset}.csv' complete. Output written to '{saveName}.csv'.")
    
    
def check_prompt_performance_for_topic(
    topicId,
    dataset,
    constrainedOutput=True,
    groundTruthCol=None
):
    """
    Evaluates how well the single topic (identified by 'topicId') performs on
    'dataset.csv' (semicolon-separated), always using ground truth data.

    CSV layout assumption:
      - The text to classify is in column 0.
      - Each topic's ground truth is at column: (topic_index * 2) + 1
        unless 'groundTruthCol' is explicitly provided.

    Steps:
      1) Locate the topic by ID in the global 'topics' list.
      2) Determine the topic's index (i.e., 0-based) among 'topics'.
      3) groundTruthCol defaults to (topic_index * 2) + 1 if not set.
      4) For each non-header row:
         - The text is row[0].
         - The ground truth is row[groundTruthCol].
         - If ground truth is empty, skip.
         - Construct the prompt from the topic's prompt (replacing [TOPIC], [CATEGORIES], [TEXT]).
         - Call getAnswerSingleTopic(...) to get a predicted category.
         - Count correct predictions for accuracy.
      5) Print the final accuracy or "No relevant attempts" if none.

    :param topicId: The ID of the desired topic (e.g. 'A' or 'B').
    :param dataset: The base name (without extension) of the CSV file.
    :param constrainedOutput: If True, pass True to getAnswerSingleTopic for constrained output.
    :param groundTruthCol: An integer column index for the ground truth; defaults to
                           (topic_index * 2) + 1 if None.
    """
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return

    # 1) Find the topic by ID and its index (topic_index)
    found_topic = None
    topic_index = None
    for i, t in enumerate(topics):
        if t.get('id') == topicId:
            found_topic = t
            topic_index = i
            break

    if found_topic is None:
        print(f"No topic found with ID={topicId}.")
        return

    # 2) If groundTruthCol is not provided, compute default from topic_index
    if groundTruthCol is None:
        groundTruthCol = (topic_index * 2) + 1

    # 3) Build a list of category strings for getAnswerSingleTopic
    local_categories = [
        cat_input.value
        for (cat_input, _, cat_id) in found_topic.get('categories', [])
    ]

    # We'll track how many attempts we made and how many were correct
    relevant_attempts = 0
    correct_predictions = 0

    # 4) Read the CSV
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)

    for rowIndex, row in enumerate(rows):
        # Skip the header row
        if rowIndex == 0:
            continue

        # We need at least groundTruthCol columns in this row
        if len(row) <= groundTruthCol:
            continue

        text_to_classify = row[0].strip()
        groundTruthCategoryName = row[groundTruthCol].strip()

        # Skip empty groundTruth
        if not groundTruthCategoryName:
            continue

        # Construct the final prompt from the topic's prompt
        prompt_template = found_topic['prompt'].value
        prompt_categories_str = "[" + ",".join(f"'{cat}'" for cat in local_categories) + "]"

        # Replace placeholders
        prompt = prompt_template.replace('[TOPIC]', found_topic['topic_input'].value)
        prompt = prompt.replace('[CATEGORIES]', prompt_categories_str)
        prompt = prompt.replace('[TEXT]', text_to_classify)

        # Classify using getAnswerSingleTopic
        answer = getAnswerSingleTopic(prompt, local_categories, constrainedOutput)

        # Debug print if desired:
        # print(f"Row {rowIndex}: Predicted: {answer} | GroundTruth: {groundTruthCategoryName}")

        # Update attempts & correctness
        relevant_attempts += 1
        if answer == groundTruthCategoryName:
            correct_predictions += 1

    # 5) Print the final result
    if relevant_attempts > 0:
        accuracy = (correct_predictions / relevant_attempts) * 100.0
        print(f"Topic (ID={topicId}) => Accuracy: {accuracy:.2f}%  "
              f"({correct_predictions} / {relevant_attempts} attempts)")
    else:
        print(f"Topic (ID={topicId}): No relevant attempts (no rows with non-empty groundTruth).")
        

        
        
def getLLMImprovedPromptWithFeedback(old_prompt, old_accuracy, topic_info):
    """
    Uses an LLM to improve 'old_prompt' given 'old_accuracy'.
    Incorporates the topic name and available categories as context.
    Returns a new prompt string (or the old prompt if something fails).
    """
    # 1) Gather context from topic_info
    topic_name = topic_info['topic_input'].value
    category_list = [cat_input.value for (cat_input, _, _cat_id) in topic_info['categories']]
    category_str = ", ".join(category_list) if category_list else "No categories defined"

    # 2) Prepare system/user content for an LLM call
    system_content = (
        f"You are an advanced prompt engineer.\n"
        f"The classification topic is '{topic_name}'.\n"
        f"The available categories for this topic are: {category_str}\n"
        "Rewrite the user's prompt to achieve higher accuracy on classification tasks.\n"
        "You MUST keep the placeholder [TEXT].\n"
        "IMPORTANT: Output ONLY the final prompt, wrapped in triple backticks.\n"
        "No commentary, bullet points, or explanations.\n"
        "The new prompt should be in English.\n"
    )

    user_content = (
        f"Previously, the prompt achieved an accuracy of {old_accuracy:.2f}%. \n"
        "Here is the old prompt:\n\n"
        f"{old_prompt}\n\n"
        "Please rewrite/improve this prompt. Keep [TEXT]. "
        "Wrap your entire revised prompt in triple backticks, with no extra lines."
    )

    # 3) Distinguish between OpenAI/DeepInfra vs local model approach
    if promptModelType in ("OpenAI", "DeepInfra"):
        try:
            completion = client.chat.completions.create(
                model=promptModel,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=250,
                temperature=0.7
            )
            improved_prompt = completion.choices[0].message.content.strip()

            # Extract the text inside ```...```
            match = re.search(r"```(.*?)```", improved_prompt, flags=re.DOTALL)
            if match:
                improved_prompt = match.group(1).strip()
            else:
                print("Warning: The LLM did not provide triple backticks. Using full text.")

            print("Improved Prompt:", improved_prompt)  # Debug

            if not improved_prompt or "[TEXT]" not in improved_prompt:
                print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                return old_prompt

            return improved_prompt

        except Exception as e:
            print(f"Error calling OpenAI/DeepInfra: {e}")
            return old_prompt

    else:
        # Local approach (Guidance or custom model)
        try:
            base_instruction = system_content
            improvement_request = (
                f"{base_instruction}\n\n"
                f"Original prompt:\n{old_prompt}\n"
            )

            # If you have a “constrainedOutputCheckbox”-like mechanism, remove or replace it here.
            # We'll assume a simple boolean 'constrainedOutput' you can pass in if needed.
            # For demonstration, we'll skip it and do a simple generation:

            script = promptModelGuidance + f" {improvement_request}" + gen(max_tokens=250, name='improvedPrompt')
            new_prompt = script["improvedPrompt"]

            if not new_prompt or "[TEXT]" not in new_prompt:
                print("Warning: The improved prompt is empty or lacks [TEXT]. Reverting to old prompt.")
                return old_prompt

            return new_prompt

        except Exception as e:
            print(f"Error calling local approach: {e}")
            return old_prompt
        

        
        



        
def evaluate_prompt_accuracy(topic_info, prompt, dataset, constrainedOutput, groundTruthCol):
    """
    Evaluates the accuracy of a given prompt for a specific topic on a dataset.
    
    :param topic_info: The topic dictionary.
    :param prompt: The prompt to evaluate.
    :param dataset: Base name of the CSV file (without .csv).
    :param constrainedOutput: Passed to getAnswerSingleTopic.
    :param groundTruthCol: Column index for ground truth in the CSV.
    :return: Accuracy as a percentage (float).
    """
    csv_file = dataset + ".csv"
    if not os.path.exists(csv_file):
        print(f"No {csv_file} file found.")
        return 0.0

    local_categories = [cat_input.value for (cat_input, _, _) in topic_info.get('categories', [])]
    relevant_attempts = 0
    correct_predictions = 0

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)

    for i, row in enumerate(rows):
        if i == 0:  # skip header
            continue
        if len(row) <= groundTruthCol:
            continue

        text_to_classify = row[0].strip()
        groundTruthCategoryName = row[groundTruthCol].strip()
        if not groundTruthCategoryName:
            continue

        prompt_categories_str = "[" + ",".join(f"'{cat}'" for cat in local_categories) + "]"
        final_prompt = prompt.replace('[TOPIC]', topic_info['topic_input'].value)
        final_prompt = final_prompt.replace('[CATEGORIES]', prompt_categories_str)
        final_prompt = final_prompt.replace('[TEXT]', text_to_classify)

        answer = getAnswerSingleTopic(final_prompt, local_categories, constrainedOutput)
        relevant_attempts += 1
        if answer == groundTruthCategoryName:
            correct_predictions += 1

    if relevant_attempts > 0:
        return (correct_predictions / relevant_attempts) * 100.0
    return 0.0    

        
def improve_prompt(topicId, dataset, constrainedOutput=True, groundTruthCol=None, num_iterations=10):
    """
    Improves the prompt for a given topic using LLM feedback and dataset evaluation.
    
    :param topicId: The ID of the topic to improve (e.g., 'A').
    :param dataset: Base name of the CSV file (without .csv).
    :param constrainedOutput: Whether to use constrained output for classification.
    :param groundTruthCol: Column index for ground truth; defaults based on topic position.
    :param num_iterations: Number of improvement iterations.
    """
    # Find topic by ID
    found_topic = next((t for t in topics if t.get('id') == topicId), None)
    if not found_topic:
        print(f"No topic found with ID {topicId}.")
        return

    # Determine topic index and default groundTruthCol
    topic_index = topics.index(found_topic)
    if groundTruthCol is None:
        groundTruthCol = (topic_index * 2) + 1

    # Evaluate baseline accuracy for the current prompt
    old_prompt = found_topic['prompt'].value
    old_accuracy = evaluate_prompt_accuracy(found_topic, old_prompt, dataset, constrainedOutput, groundTruthCol)

    best_prompt = old_prompt
    best_accuracy = old_accuracy

    print("========================================")
    print(f"Starting iterative prompt improvement for topic '{found_topic['id']}'")
    print(f"Baseline accuracy: {best_accuracy:.2f}%")
    print("========================================")

    for iteration in range(1, num_iterations + 1):
        new_prompt = getLLMImprovedPromptWithFeedback(best_prompt, best_accuracy, found_topic)
        if "[TEXT]" not in new_prompt:
            print("Warning: The improved prompt lost [TEXT]. Skipping iteration.")
            continue

        new_accuracy = evaluate_prompt_accuracy(found_topic, new_prompt, dataset, constrainedOutput, groundTruthCol)
        diff = new_accuracy - best_accuracy

        print(f"Iteration {iteration}:")
        print(f"New prompt accuracy: {new_accuracy:.2f}% (was {best_accuracy:.2f}%)")

        if diff > 0.001:
            print(f"Improvement found (+{diff:.2f}%). Updating best prompt.")
            best_prompt = new_prompt
            best_accuracy = new_accuracy
        else:
            print("No improvement. Keeping current best prompt.")
        print("----------------------------------------")

    print("========================================")
    print(f"Final best accuracy: {best_accuracy:.2f}%")
    print("Best prompt:\n", best_prompt)
    print("========================================\n")

    # Store the best prompt and its accuracy in the topic if improved
    if best_accuracy > old_accuracy:
        found_topic['best_prompt_found'] = best_prompt
        found_topic['best_prompt_accuracy'] = best_accuracy
    else:
        found_topic['best_prompt_found'] = None
        found_topic['best_prompt_accuracy'] = None
                
        
class MockText:
    def __init__(self, value: str):
        self.value = value