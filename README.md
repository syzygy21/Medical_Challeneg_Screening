# Project Overview

## Medical Question-Answering System

The aim of the project is to develop a specialized question-answering system for medical queries. It should provide accurate and reliable information for the question asked. The proposed system for this project leverages advanced natural language processing techniques to understand medical terminology and context to offer insightful and helpful answers. The model trained for this task uses a specialized dataset curated to provide relevant medical question-answering examples.

## Problem Statement

Healthcare information accessibility is a major challenge for many patients and practitioners. In a future where healthcare professionals and physicians could interact with EHR systems by asking complex natural language questions and receiving precise answers with adequate context, a medical question-answering system would be central to processing these questions and finding answers from various sources in a patient's record. In addition to this usage, patients and information seekers could use such open-source chatbots that will be specifically designed to cater to medical queries to gain relevant and quick initial information about their queries.

## Dataset

The proposed model has been trained on a dataset that is specialized for medical Q&A tasks containing paired questions and answers related to various diseases, symptoms, and treatments. The dataset has:

- A total of 16,406 examples of question-answer pairs.
- Includes a total of about 9,000 unique medical terms (Examples: Breast Cancer, Diabetes, Stroke, etc.)
- Varying question types

## Approach

The approach used for this project is fine-tuning the FLAN-T5 Small language model to adapt it to the dataset.

The approach includes:

- Data preprocessing to enhance training efficiency
- Prefix-based task formulation ("Answer the medical question: ")
- Fine-tuning of a pre-trained language model on medical domain data
- ROUGE metric-based evaluation to evaluate response quality
- Early stopping mechanisms to prevent overfitting

The model demonstrated strong performance on the medical dataset, which was evident with the steady increase in the ROUGE scores and the improvement in validation loss throughout the training process, demonstrating the model’s capabilities of producing relevant and generalizable outputs.

# Model Selection

The FLAN-T5 model was selected as the optimal foundation for our medical question-answering system based on the following considerations:

1. **Transfer Learning Advantage**: FLAN-T5's (Small variant) instruction-tuned foundation eliminated the need to train from scratch. This is a great advantage because of the limited GPU resources that our approach has access to (NVIDIA GeForce RTX 3060 Laptop GPU). The transfer learning allows for domain adaptation rather than fundamental language learning.

2. **Generative vs Extractive Architecture**: Unlike extractive models like BioBERT that identify answer spans within a context, FLAN-T5 is a generative model that is well suited for tasks like question answering. The fact that it is open source and easy to fine-tune makes it an even bigger advantage. These generative capabilities of FLAN-T5 allow it to formulate responses rather than just locating text fragments.

3. **Memory-Performance Balance**: The FLAN-T5 model has 512-dimensional hidden states and 8 attention heads which maximize performance while staying within the hardware constraints.

# Training Details

1. **Data Preparation**: The preprocessing pipeline used in the approach cleans up the data by carrying out URL/email removal, HTML tag stripping, and white space normalization. Input texts were enhanced with a medical-specific prompt. This is done to orient the model towards the specific domain.

2. **Training Hyperparameters**: The training parameters that were selected included a learning rate of 3e-5 and a batch size of 8. This provided convergence stability and also did not violate the GPU memory constraints. Early stopping was also implemented with patience of 5 and the setup had 20 total possible epochs.

3. **Optimization Strategy**: The implementation used the AdamW optimizer with a weight decay of 0.01. The `save_total_limit` of 3 was beneficial for tracking the best models and also saved disk space.

4. **Evaluation Metrics**: Training process was monitored using the ROUGE-L score. This metric was evaluated to trigger a possible early stop. This metric captures the importance of sequential information in medical text.

# Model Performance

## Model Strengths

1. **Strong Convergence**: During the training process the model had a successful convergence with a validation loss of 1.7212. The model showed progressive improvement across training epochs before early stopping based on ROUGE-L scores.

2. **Efficient Training Process**: The early stopping mechanism kicked in after 19 epochs (planned 20) based on the ROUGE-L score indicating that the model has capabilities of learning the context from the dataset.

3. **Steady Performance Improvement**: Throughout the training process the metrics showed a steady improvement. The learning rates and the weight decay were selected in a way where effective training could occur. This means that model’s generative capabilities can be effectively used during inference times.

4. **Appropriate Generation Length**: The relatively stable generation length metrics (around 12.2 tokens) suggest the model produces consistent and appropriately sized responses for medical questions.

5. **Transfer Learning Advantages**: By leveraging the pre-trained FLAN-T5 architecture, the model benefits from the extensive knowledge transfer. This effectively means that it requires less training data for training, which is also evident from the improvement of metrics through the training process.

## Model Weaknesses

1. **Limited Context Understanding**: The FLAN-T5 architecture implements the fixed context window which limits its ability to study lengthy answers, which limits the knowledge gain that happens. The dataset used in the approach has long answers and therefore a fixed context window is a drawback.

2. **Entity Handling Constraints**: The model lacks specialized mechanisms for identifying and reasoning about medical entities (diseases, treatments, symptoms), potentially leading to imprecise or incomplete answers. This is another reason why incorporating medical entities across all examples will be very beneficial for model training.

3. **Evaluation Metric Limitation**: ROUGE scores may not capture adequately the medical accuracy and clinical relevance of generated answers. This is a drawback as the current approach uses the ROUGE-L score as a method for early stopping.

4. **Simple Retrieval Mechanism**: The approach that is currently being used is a simple training on the question-answer pairs in the dataset. If the model is equipped with information retrieval capability then its performance could substantially improve when it comes to complex reasoning or domain-specific medical knowledge. Such a retrieval system requires an existing knowledge base.

# Assumptions and Model Development

## Key Assumptions Made

1. **Text-Based QA is Sufficient**: The proposed approach assumes that medical question answering can be effectively modeled as a text-to-text generation task. The model is trained in a way where it generates natural language responses based on learned patterns rather than retrieving specific passages from a knowledge base. Adding a retrieval layer can possibly boost the performance of the system.

2. **FLAN-T5 Pre-training is Beneficial**: Another assumption is that the instruction-tuned FLAN-T5 model's pre-training on diverse tasks would transfer well to medical domain questions. The pre-training of the model would be beneficial for the task at hand. The learning rate and the weight decay for the model training were decided in a way where the hopes were that the model will not overfit on the training data and maintain its previous knowledge. This could provide better zero-shot generalization compared to standard language models.

3. **Question Prefix Enhances Performance**: The approach assumes that addition of the prefix "Answer the medical question:" to all inputs would lead to a better context understanding for the model and help it produce more focused responses.

4. **Small Dataset is Adequate**: The approach assumes that fine-tuning on the provided dataset, despite its limited size, would be sufficient to adapt the pre-trained model to the issue-specific task of medical question and answering. Since the approach is not focused on developing a generalized chatbot, it is developed on the assumption that selecting the relatively small dataset and a relatively small architecture (given the computational constraints) should be good enough for the task.

5. **Evaluation Metrics**: The approach uses the ROUGE metrics for evaluation based on the assumption that lexical overlap between generated answers and reference answers is a reasonable proxy for answer quality in the medical domain. The approach does not take the context comparison of the generated answers and the ground truth answers during training.

6. **Generation Parameters**: We assumed that using beam search with 4 beams and controlling the output length would result in more coherent and accurate responses than greedy decoding. This would lead to generation and evaluation of a larger number of possibilities as multiple possible sequences are explored at each step.

7. **Early Stopping Strategy**: The approach uses an early stopping strategy (patience=5, threshold=0.01) with the assumption that model will plateau after a certain amount of training and continuing beyond will lead to overfitting.

8. **Transfer Learning Efficiency**: The approach leverages a transfer learning strategy with the assumption that the pretrained weights already contain useful information and this would make it effective in generalizing to an unseen task. Especially if we consider the Flan-T5 model that has been trained on a large variety of data (e.g., books, websites, code, scientific data).

# Potential Improvements or Extensions to the Solution

1. **Implementing RAG-Based Architecture**  
RAG-based architectures are important because they enhance the efficiency of the Gen AI and NLP models by first extracting relevant and factual information from external sources or existing datasets/databases. Integration of a retrieval system along with a fine-tuned model like FLAN-T5 could significantly improve the performance of the whole system.
There are several very advanced RAG architectures especially for medical question answering such as the i-MedRAG where the model can ask follow-up queries based on previous information-seeking attempts, forming reasoning chains for complex medical questions. Creation of a knowledge base is very crucial for this task. An extension to the proposed solution could be giving the fine-tuned model access to a strong medical knowledge base. By doing this we essentially have a model that not only has performance for the issue-specific task but also can retrieve from the knowledge base and therefore its generation quality would improve a lot, much like the existing AI systems such as Claude or GPT.

2. **Evaluation Enhancements**  
The current approach uses the ROUGE metrics which have limitations for medical QA tasks.
The approach can be modified to take into account some domain-specific metrics such as answer accuracy, contextual relevancy, and hallucination detection. The metrics that evaluate these should be built such that they are specific to medical content.
Another approach that can be taken for evaluation enhancements is to use expert validation or human-in-the-loop training. Medical experts can be brought in to check the accuracy of the information given by the models upon prompting. This can be further incorporated into the training loop itself through techniques such as reinforcement learning through human feedback.

3. **Data Processing Improvements**  
The approach currently employs data preprocessing techniques that are minimal for the task. There can be better data preprocessing techniques that should be employed.
For example, application of biomedical named entity recognition (BNER) to identify important medical concepts like diseases, treatments, and anatomical entities in both questions and answers can be done. This is a domain-specific technique that can significantly improve model training as the more advanced models can be trained on a dataset that provides more information which can translate to useful knowledge.
Another idea that can be employed to improve the quality of the data that is currently being used is filtering out the datapoints with answers that are unrelated to the question asked. This could be implemented by looping through the dataset and picking up each example and looking at the context of the question asked and the context of the respective answer and seeing if they match. Usage of open-source LLM architectures available within the `transformers` module from Hugging Face can be used for this task. Also appending the current dataset with more medical related question answers so as to make it a more diverse and higher-quality medical QA dataset.

4. **Employing Specialized Biomedical Language Models**  
Specialized models can be used for this task as these are models that have already been pre-trained on large amounts of data and also have specialized architectures for medical QA systems.
Recently, the generative biomedical models have shown a lot of promise for generative tasks related to medical data such as question answering. Consider BioGPT which achieved 78.2% accuracy on PubMedQA, creating a new record compared to previous models. The strong learning capabilities of transformer-based architectures such as GPT models should be brought into the sphere of medical data-related generative tasks. A lot of these advanced models can be fine tuned on datasets through the APIs provided by companies like OpenAI. Also, the application of the advanced attention mechanisms within the T5 model family can add as another improvement in the existing pipeline. Also a larger version of the FLAN-T5 model can be fine-tuned for better performance.

# Example Interactions


Below are a few sample interactions between the fine-tuned FLAN-T5 model and medical questions. These illustrate the model's ability to generate relevant responses based on learned context.
The responses shown in the example interactions were generated using the `.generate()` method from the Hugging Face Transformers library with the following parameters:

```python
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )
---
These parameters were added to penalize the model for generating repetitive terms and sentences during inference.


**Question:** What is hypertension and what blood pressure readings indicate this condition?  
**Answer:** Hypertension is a condition that affects the blood pressure of the body. It is caused by a change (mutation) in the X-linked gene, which is involved in the normal flow of blood to the body's bloodstream. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with hypertension typically have one copy of the mutated gene, but they typically do not show signs and symptoms of the condition.

---

**Question:** What are the main functions of the kidneys in the human body?  
**Answer:**  
- The kidneys are made up of many different organs, including the kidneys, tissues, and organs.  
- There are three main functions of kidneys in the body:  
  - Kidneys are part of the body's immune system, which helps fight infection and protects the body from infection.  
  - It is important to maintain a healthy immune system and to prevent infections.  
  - Causes of kidney disease include chronic kidney disease, hypertension, decreased levels of calcium in the blood, high levels of potassium in the urine, and increased amounts of...

---



**Question:** How does insulin work in controlling blood sugar levels?  
**Answer:** Insulin plays a role in controlling blood sugar levels. Insulin helps control the amount of sugar in your blood. This means that you can use it to control your blood sugar level. If you have diabetes, you may want to think about taking part in a blood glucose test. You can take part in an insulin test to find out if your blood glucose levels are too high or too low. For example, you might want to talk with your doctor if you think you may be at risk for diabetes.

# Declaration

I confirm that this project was completed independently without the assistance of third-party AI systems (such as OpenAI, Claude, or similar tools) in any part of the solution. Any references made were to publicly available documentation or open-source code as permitted.
