# Infinity Instruct

<p align="center">
<img src="static/Bk3NbjnJko51MTx1ZCScT2sqnGg.png" width="300">
</p>
<p align="center">
<em>Beijing Academy of Artificial Intelligence (BAAI)</em><br/>
<em>[Paper][Code][🤗] (would be released soon)</em>
</p>

The quality and scale of instruction data are crucial for model performance. Recently, open-source models have increasingly relied on fine-tuning datasets comprising millions of instances, necessitating both high quality and large scale. However, the open-source community has long been constrained by the high costs associated with building such extensive and high-quality instruction fine-tuning datasets, which has limited related research and applications. To address this gap, we are introducing the **Infinity Instruct** project, aiming to develop a large-scale, high-quality instruction dataset.

## **News**

- 🔥🔥🔥[2024/06/13] We are pleased to share the results of our intermediate checkpoints. Our ongoing efforts focus on risk assessment and data generation. The finalized version with 10 million instructions is scheduled for release in late June.

Flopsera [[http://open.flopsera.com/flopsera-open/details/InfinityInstruct](http://open.flopsera.com/flopsera-open/details/InfinityInstruct)]

huggingface[[https://huggingface.co/datasets/BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)]

## **GPT-4 automatic evaluation**

|           **Model**          	| **MT-Bench** 	| **AlpacaEval2.0** 	|
|:----------------------------:	|:------------:	|:-----------------:	|
| OpenHermes-2.5-Mistral-7B    	|      7.5     	|        16.2       	|
| Mistral-7B-Instruct-v0.2     	|      7.6     	|        17.1       	|
| Llama-3-8B-Instruct*         	|      8.1     	|        22.9       	|
| GPT 3.5 Turbo 0613           	|      8.4     	|        22.7       	|
| Mixtral 8x7B v0.1            	|      8.3     	|        23.7       	|
| Gemini Pro                   	|      --      	|        24.4       	|
| InfInstruct-3M-Mistral-7B   	|      7.3     	|        14.3       	|
| InfInstruct-Mistral-7B 0608  	|      7.8     	|        16.9       	|
| InfInstruct-Mistral-7B 0612  	|      7.9    	|        25.1       	|
| GPT-4-0613                   	|      9.2     	|        30.2       	|
| Llama-3-70B-Instruct*        	|      9.0     	|        34.4       	|
| InfInstruct-3M-Llama-3-70B  	|      8.4     	|        21.8       	|
| InfInstruct-Llama-3-70B 0608 	|      8.9     	|        27.1       	|
| InfInstruct-Llama-3-70B 0612 	|      8.6     	|        30.7       	|
| InfInstruct-Llama-3-70B 0613 	|      8.7     	|        31.5        	|

*denotes the results come from [web](https://huggingface.co/abacusai)

## Performance on **Downstream tasks**

|          **Model**          	|  **MMLU** 	| **GSM8K** 	| **HumanEval** 	| **HellaSwag** 	| **Average** 	|
|:---------------------------:	|:---------:	|:---------:	|:-------------:	|:--------------:	|:-----------:	|
| Mistral-7B                  	|    56.5   	|    48.1   	|      14.0     	|      35.5      	|     38.5    	|
| Mistral-7B-Instruct-v0.2    	|    59.6   	|    45.9   	|      32.9     	|      64.4      	|     50.7    	|
| OpenHermes-2.5-Mistral-7B   	|    61.7   	|    73.0   	|      41.5     	|      80.6      	|     64.2    	|
| InfInstruct-3M-Mistral-7B   	|    62.9   	|    78.1   	|      50.6     	|      84.8      	|     69.1    	|

## Overview of Infinity Instruct
![](static/whiteboard_exported_image.png)

## Data sources

We collect large-scale instruct data from the open-source community. The data sources are listed as follows:

- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft)
- [CodeBagel](https://huggingface.co/datasets/Replete-AI/code_bagel)
- [CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/Code-Feedback)
- [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)
- [CodeExercise-Python-27k](https://huggingface.co/datasets/codefuse-ai/CodeExercise-Python-27k)
- [Evol-Instruct-Code-80k-v1](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1)
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)
- [MetaMathQa](https://huggingface.co/datasets/meta-math/MetaMathQA)

- The detailed information on Infinity-Instruct-3M is shown in the following table.
  
| **Raw Dataset**                               	| **Numbers of Rows** 	|
|-----------------------------------------------	|:-------------------:	|
| glaiveai/glaive-code-assistant-v3             	|        138157       	|
| Replete-AI/code_bagel_hermes-2.5              	|        506346       	|
| m-a-p/CodeFeedback-Filtered-Instruction       	|        104848       	|
| bigcode/self-oss-instruct-sc2-exec-filter-50k 	|        50661        	|
| codefuse-ai/CodeExercise-Python-27k           	|        27224        	|
| nickrosh/Evol-Instruct-Code-80k-v1            	|        78264        	|
| TIGER-Lab/MathInstruct                        	|        188486       	|
| microsoft/orca-math-word-problems-200k        	|        200035       	|
| MetaMathQa                                    	|        395000       	|
| teknium/Openhermes-2.5                        	|       1001551       	|
| Math                                          	|        320130       	|
| Selected subjective instructions              	|       1362000       	|
| **Summary**                                     |     **4372702**       |


- Source and number of subjective instructions:

| **Raw Dataset**              	| **Numbers of Rows** 	|
|------------------------------	|:-------------------:	|
| Alpaca GPT4 data             	|        13490        	|
| Alpaca GPT4 data zh          	|        32589        	|
| Baize                        	|        14906        	|
| BELLE Generated Chat         	|        43775        	|
| BELLE Multiturn Chat         	|        210685       	|
| BELLE 3.5M CN                	|        312598       	|
| databricks-dolly-15K         	|        10307        	|
| LIMA-sft                     	|         712         	|
| CodeContest                  	|         523         	|
| LongForm                     	|         3290        	|
| ShareGPT-Chinese-English-90k 	|         8919        	|
| UltraChat                    	|        276345       	|
| Wizard evol instruct zh      	|        44738        	|
| Wizard evol instruct 196K    	|        88681        	|
| BELLE School Math            	|        38329        	|
| Code Alpaca 20K              	|        13296        	|
| WildChat                     	|        61873        	|
| COIG-CQIA                    	|        45793        	|
| BAGEL                        	|        55193        	|
| DEITA                        	|        10000        	|
| Math                         	|        320130       	|
| **Summary**                  	|     **1362000**     	|

The domain distribution of the subjective instruction category are shown in the following picture.

![](static/PX0ybsIyUoCy3rxgjEzcrFTnnPg.png)

## **Instruction Selection for downstream tasks**

To create an objective ranking, we utilize datasets such as Flan and OpenHermes, with a focus on enhancing code and math capabilities. The method includes detailed topic distribution tagging of the evaluation set (e.g., data structures, sorting in humaneval). We apply heuristic rules to filter out irrelevant data based on the dataset source (e.g., removing network or file I/O operations). We further retrieve a subset from the training set based on the distribution in the validation sets.

## **Instruction ****G****eneration for ****H****igh-****Q****uality ****R****esponse**

![](static/dataflow.png)

### High-Quality Open Source Instruction Collection and Tag System

We start by collecting high-quality open-source instruction sets. We assign each instruction in the collection a set of tags that describe the abilities and knowledge necessary to complete the instruction. With this tagging system, we can recognize the content distribution of the collection and the abilities required for completing different tasks.

- Instruction collection: We systematically reviewed available open-source instruction sets and included sets created by humans and advanced LLMs.
- Tag System: with totally two levels:

  - First level tag: Describe the specific knowledge and abilities required for completing each instruction (e.g., Arithmetic Calculation, Knowledge of Biology). The tags are automatically generated by LLM.
  - Second level tags： Macro categories such as "Natural Language Processing" and "Math Reasoning."  Including 25 categories in total.

### Informative Instruction Selection

Aimed at selecting most informative instructions from the whole collection for enhancing the performance of LLM and improving user experience.

- Informative Instructions:
  - Instructions demand multiple kinds of abilities or multiple domains of knowledge. Such instructions are recognized by our tag system.
  - Instructions with long-tailed ability or knowledge;
  - Instructions with high following difficulty. The following difficulty of instructions is obtained using the method of Li et al. [1].

### Instruction Generation by Data Evolution Strategy

We expand the seed instructions in directions breadth, depth, difficulty, and complexity with a method built based on [2], and use AI assistants to generate multi-turn data.

- Based on the metadata selected in the previous section, we expand the instructions by randomly selecting one dimension from breadth, depth, difficulty and complexity dimensions on the basis of the Evol-Instruct method.
- Validate the evolved data, and use AI assistants to eliminate data that failed to evolve from the perspective of instruction compliance.
- Use the evolved instructions as the initial input, and use an AI assistant to play different roles to generate 2 to 4 rounds of dialogue for each instruction.

### Instruction Generation by Model Ability Deficient Diagnosis

Automatically identifying weaknesses in the model's capabilities to guide the synthesis of data.

- Model performance evaluation System: Constituted by a collection of commonly used evaluation sets;
- Automatic ability deficient diagnosis: Inducing shortcuts based on ground truth answers and model outputs using AI assistants;
- Targeted data synthesis: Automatically generate new instructions using AI assistants based on the induced deficiencies.

## **Disclaimer**

The resources, including code, data, and model weights, associated with this project are restricted for academic research purposes only and cannot be used for commercial purposes. The content produced by any version of Infinity Instruct is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.

## 

## Reference

[1] Li M, Zhang Y, He S, et al. Superfiltering: Weak-to-strong data filtering for fast instruction-tuning[J]. arXiv preprint arXiv:2402.00530, 2024.

[2] Xu C, Sun Q, Zheng K, et al. WizardLM: Empowering large pre-trained language models to follow complex instructions[C]//The Twelfth International Conference on Learning Representations. 2023.

## Citation
Our paper, detailing the development and features of the **Infinity Instruct** dataset, will be released soon on arXiv. Stay tuned!

```
@article{InfinityInstruct2024,
  title={Infinity Instruct},
  author={Beijing Academy of Artificial Intelligence (BAAI)},
  journal={arXiv preprint arXiv:2406.XXXX},
  year={2024}
}
```
