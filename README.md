# MORE
This is the official project website for the paper [Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective](https://arxiv.org/pdf/2403.18346.pdf).

## Pipeline
![Our framework for generating data of MORE](https://github.com/OpenCausaLab/MORE/blob/b71f67529e5bc8fb5ef91665d85b6895d40a937f/static/images/fig-more-generation.png)


## Dataset
### Comparison
| Datasets         | Knowledge-based | Multi-hop Reasoning | Answer Type | Unimodal Biases Evaluation | Rationale | # Size |
| ---------------- | --------------- | ------------------- | ----------- | -------------------------- | --------- | ------ |
| Visual7W (Zhu et al., 2016) | ❌              | ❌                   | Open-ended  | ❌                        | ❌       | 327.9K |
| VQA (v2) (Goyal et al., 2017) | ❌              | ❌                   | Open-ended  | ❌                        | ❌       | 1.1M   |
| FVQA (Wang et al., 2017) | ✅              | ❌                   | Open-ended  | ❌                        | ✅       | 5.8K   |
| OKVQA (Marino et al., 2019) | ✅              | ❌                   | Open-ended  | ❌                        | ❌       | 14K    |
| S3VQA (Jain et al., 2021) | ✅              | ❌                   | Open-ended  | ❌                        | ❌       | 7.5K   |
| A-OKVQA (Schwenk et al., 2022) | ✅              | ❌                   | Multi-choice | ❌                        | ✅       | 23.7K  |
| INFOSEEK (Chen et al., 2023) | ✅              | ❌                   | Open-ended  | ❌                        | ❌       | 1.4M   |
| MORE (Ours)       | ✅              | ✅                   | Multi-choice | ✅                        | ✅       | 12K    |

Table 1: Comparison of MORE with other VQA datasets, highlighting its incorporation of external knowledge, multi-hop reasoning, unimodal bias evaluation, and rationale for interpretability.



### Data Format
#### Train / Val
```JSON5
{
        "data_id": "more_val_0",
        "image_id": "oven_05009956",
        "entity": [
            "Q165765",
            "Dornier Flugzeugwerke"
        ],
        "hop": 2,
        "question": "Where is the headquarters location of the parent organization of this aircraft?",
        "direct_answers": [
            "Untert\u00fcrkheim",
            "Stoccarda",
//            ...
        ],
        "options": [
            "Toulouse, France",
            "stuttgard",
            "manzell",
            "Dornier Flugzeugwerke"
        ],
        "correct_option_idx": 1,
        "vision_option": [
            "Dornier GmbH",
            "Dornier-Werke",
//            ...
        ],
        "language_option": "Toulouse, France",
        "semantic_misleading_option": [
            "manzell",
            "kluftern",
//            ...
        ],
        "rationale": "To answer the question, first, I need to identify what this aircraft is. From the image, this aircraft is Dornier Flugzeugwerke. Then, I need to infer the parent organization of Dornier Flugzeugwerke, which is Daimler Benz. Then, I need to infer the headquarters location of Daimler Benz, which is stuttgard. Therefore, the answer is: stuttgard."
    }
```
#### Test
```JSON5
{
        "data_id": "more_test_0",
        "image_id": "oven_04953332",
        "question": "Which body of water is located in or next to the place where the architect of this building died?",
        "options": [
            "Palace of Justice, Bucharest",
            "izvorul oticului river",
            "river seine",
            "Lake Zurich"
        ]
}
```

### Statistics
| Dataset        | #I, Q, A      | Len of Q / A  | # Ent |
| -------------- | ------------- | ------------- | ----- |
| MORE-train     | 10K           | 14.3 / 2.1    | 1,261 |
| - 2-hop        | 4,134         | 11.6 / 2.0    | 886   |
| - 3-hop        | 5,866         | 16.1 / 2.2    | 686   |
| MORE-dev       | 1K            | 13.8 / 2.3    | 118   |
| - 2-hop        | 548           | 12.2 / 2.2    | 71    |
| - 3-hop        | 452           | 15.8 / 2.5    | 73    |
| MORE-test      | 1K            | 13.9 / 2.4    | 251   |
| - 2-hop        | 500           | 12.3 / 2.2    | 153   |
| - 3-hop        | 500           | 15.6 / 2.6    | 143   |

Table 2: Dataset statistics of different hops.



## Citation
Please cite our paper if this repository inspires your work.
```bibtex
@article{chen2024quantifying,
  title={Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective},
  author={Chen, Meiqi and Cao, Yixin and Zhang, Yan and Lu, Chaochao},
  journal={arXiv preprint arXiv:2403.18346},
  year={2024}
}
```

# Contact 
- meiqichen@stu.pku.edu.cn
- caoyixin2011@gmail.com
- zhyzhy001@pku.edu.cn
- luchaochao@pjlab.org.cn
