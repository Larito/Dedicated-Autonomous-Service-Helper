from transformers import BertTokenizer, BertForSequenceClassification
import torch
import subprocess
import time


def stt():
    command = "cheetah_demo_mic --access_key ndHk4eU4SVLK0IjTP/a29PaK0k50Ukst396Hk1MteSDP+a5230JZag=="
    
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("We're being nosey rn")
        time.sleep(10)
        process.terminate()
        process.wait()
        stdout, stderr = process.communicate()

        stdout = str(stdout.decode('utf-8'))
        stdout = stdout.replace("Cheetah version : 1.1.0\n","").replace("Listening... (press Ctrl+C to stop)\n","").replace("\n"," ")
        
        print(stdout)
        return [stdout]

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    return []   



# Function to get predictions for a list of sentences
def predict_class(sentences):
    
    model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')  # Specify the path to the directory where you saved the fine-tuned model
    tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model/tokenizer')# Specify the path to the directory where you saved the tokenizer
    
    inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).tolist()
    return predicted_class


def get_class_map():
    return {0: "Bathroom", 1: "Elevator", 2: "Classroom", 3: "Dean's office", 4: "Library", 5: "Starbucks", 6: "Dose", 7: "Coffee shop", 8: "Jamoka", 9: "Segafredo", 10: "Robotics Lab", 11: "Prince Turki Center", 12: "Sports Center"}


if __name__ == "__main__": #main function
    sentences = stt()
    
    predicted_classes = predict_class(sentences)
    id2label = get_class_map()

    predicted_class_labels = [id2label[idx] for idx in predicted_classes]
    
    for sentence, predicted_label in zip(sentences, predicted_class_labels):
        print(f"Predicted Class: {predicted_label}")