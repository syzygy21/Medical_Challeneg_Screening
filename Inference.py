import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for potential OpenMP conflict

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def generate_answer(model, tokenizer, question, device="cuda"):
    """
    Generate an answer to a medical question using the loaded model and tokenizer.

    Args:
        model: The fine-tuned Seq2Seq model.
        tokenizer: The tokenizer corresponding to the model.
        question (str): The input medical question.
        device (str): Device to run inference on (default: 'cuda').

    Returns:
        str: The model-generated answer.
    """
    input_text = "Answer the medical question: " + question
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Medical QA Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--question", type=str, required=False, help="Medical question to ask the model")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    print(f"\n Loading model from: {args.model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Get question
    question = args.question
    if not question:
        question = input("Enter your medical question: ")

    # Generate answer
    print("\n Generating answer...")
    answer = generate_answer(model, tokenizer, question, device=device)

    # Display output
    print("\n Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
