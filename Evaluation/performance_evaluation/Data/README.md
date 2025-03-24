# Benchmarking Large Language Models in Healthcare

Over the years, Large Language Models (LLMs) have emerged as a groundbreaking technology with immense potential to revolutionize various aspects of healthcare. These models, such as [GPT-3](https://arxiv.org/abs/2005.14165), [GPT-4](https://arxiv.org/abs/2303.08774) and [Med-PaLM 2](https://arxiv.org/abs/2305.09617) have demonstrated remarkable capabilities in understanding and generating human-like text, making them valuable tools for tackling complex medical tasks and improving patient care. They have notably shown promise in various medical applications, such as medical question-answering (QA), dialogue systems, and text generation. Moreover, with the exponential growth of electronic health records (EHRs), medical literature, and patient-generated data, LLMs could help healthcare professionals extract valuable insights and make informed decisions.

However, despite the immense potential of Large Language Models (LLMs) in healthcare, there are significant and specific challenges that need to be addressed. 

When models are used for recreational conversational aspects, errors have little repercussions; this is not the case for uses in the medical domain however, where wrong explanation and answers can have severe consequences for patient care and outcomes. The accuracy and reliability of information provided by language models can be a matter of life or death, as it could potentially affect healthcare decisions, diagnosis, and treatment plans.



## Datasets, Tasks, and Evaluation Setup 

The Medical-LLM Leaderboard includes a variety of tasks, and uses accuracy as its primary evaluation metric (accuracy measures the percentage of correct answers provided by a language model across the various medical QA datasets).

### MMLU-Pro Subset (Medicine)

The [MMLU-Pro benchmark](https://arxiv.org/pdf/2406.01574)，an enhanced benchmark designed to evaluate language understanding models across broader and more challenging tasks. Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU-Pro integrates more challenging, reasoning-focused questions and increases the answer choices per question from four to ten, significantly raising the difficulty and reducing the chance of success through random guessing. MMLU-Pro comprises over 12,000 rigorously curated questions from academic exams and textbooks, spanning 14 diverse domains including Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others.


### MMLU Subset (Medicine)

The [MMLU benchmark](https://arxiv.org/abs/2009.03300) (Measuring Massive Multitask Language Understanding) includes multiple-choice questions from various domains. For the Open Medical-LLM Leaderboard, we focus on the subsets most relevant to medical knowledge:

- Clinical Knowledge: 265 questions assessing clinical knowledge and decision-making skills.
- Medical Genetics: 100 questions covering topics related to medical genetics.
- Anatomy: 135 questions evaluating the knowledge of human anatomy.
- Professional Medicine: 272 questions assessing knowledge required for medical professionals.
- College Biology: 144 questions covering college-level biology concepts.
- College Medicine: 173 questions assessing college-level medical knowledge.

Each MMLU subset consists of multiple-choice questions with 4 answer options and is designed to evaluate a model's understanding of specific medical and biological domains.

The Open Medical-LLM Leaderboard offers a robust assessment of a model's performance across various aspects of medical knowledge and reasoning.

### MedQA

The [MedQA](https://arxiv.org/abs/2009.13081) dataset consists of multiple-choice questions from the United States Medical Licensing Examination (USMLE). It covers general medical knowledge and includes 11,450 questions in the development set and 1,273 questions in the test set. Each question has 4 or 5 answer choices, and the dataset is designed to assess the medical knowledge and reasoning skills required for medical licensure in the United States.

### MedMCQA

[MedMCQA](https://proceedings.mlr.press/v174/pal22a.html) is a large-scale multiple-choice QA dataset derived from Indian medical entrance examinations (AIIMS/NEET). It covers 2.4k healthcare topics and 21 medical subjects, with over 187,000 questions in the development set and 6,100 questions in the test set. Each question has 4 answer choices and is accompanied by an explanation. MedMCQA evaluates a model's general medical knowledge and reasoning capabilities.


### PubMedQA

[PubMedQA](https://aclanthology.org/D19-1259/) is a closed-domain QA dataset, In which each question can be answered by looking at an associated context (PubMed abstract). It is consists of 1,000 expert-labeled question-answer pairs. Each question is accompanied by a PubMed abstract as context, and the task is to provide a yes/no/maybe answer based on the information in the abstract. The dataset is split into 500 questions for development and 500 for testing. PubMedQA assesses a model's ability to comprehend and reason over scientific biomedical literature.
