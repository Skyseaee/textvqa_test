## Multimodal Model Evaluation
### Overview
The current framework for multimodal model evaluation supports assessment on image datasets. The evaluation process is generally divided into the following steps:

Task Loading: Load the dataset as specified by the user, encapsulated into a Task. The loading process checks for local cache availability; if not present, it downloads from AIS (e.g., Hugging Face).

Model Deployment: Deploy the multimodal model as a service.

Inference Execution: Run the inference tasks.

Result Compilation: Aggregate results and output the corresponding evaluation metrics.

### Datasets
#### TextVQA
TextVQA is a specialized test set designed to evaluate the performance of Visual Question Answering (VQA) models in understanding and answering questions related to text within images. This test set requires models to not only comprehend visual information in images but also to recognize and understand textual content within these images, integrating both types of information to formulate answers. The introduction of TextVQA has propelled research in visual and textual multimodal understanding, challenging models to perform complex scene reasoning, and is widely applied in fields such as image understanding, OCR (Optical Character Recognition), and natural language processing.

### Usage
Repository Address: https://git.garena.com/shopee/MLP/aip/llm-sys/llvm-evalution

Run using workflow.sh.

Example Usage:

```bash
bash workflow.sh -d 0 -t 1 -m llava-v1
```

### Results Display
#### TextVQA
Currently, the primary supported evaluation metric for the TextVQA test set is accuracy, specifically represented as ACC [%].