# ReSeDis

**ReSeDis: A Benchmark for Referring-based Object Search across Large-Scale Image Collections

🔗 [**Supplementary Material**]([https://your-link-here.com](https://drive.google.com/file/d/1Xhdhx7QKX1XvxI-xbCd6-jYVL7sei5js/view?usp=sharing))


![Samples from ReSeDis](assets/rosd.png)

**ReSeDis** is a dataset for Refering Search and Discovery task to evaluate the ability of models to search for referred objects across a large image collection. It contains 7,088 images from MSCOCO and 9,664 referring expression. 

## Download Links
https://drive.google.com/drive/folders/1H0woMUkhVA0IcA8614b1oI6gajNN2g1-?usp=sharing

## Annotation
* For each object, we provide a json file. You can read the annotations using the following code:

```python
import json
with open('*.json', 'r') as f:
    infos = json.load(f)
```

* The format of annotations:
```
    <image_id>: the id of the image (same as in MSCOCO)
    <target_annotation>: segmentation mask of the target object
    <expression>: text description of the target object
    <category>: category of the target object
```

## Agreement
The ReSeDis dataset is released for non-commercial, academic research and educational purposes only.
Commercial use of the dataset is strictly prohibited.
Redistribution of the dataset or any modified versions is not allowed without written permission from the authors.
<!-- Any publications or projects using this dataset must cite the following paper:
  [Your Paper Title], [Authors], [Conference/Journal], [Year]. -->

<!-- ## Citation

If you find this dataset useful for your research and use it in your work, please consider cite the following papers:

```bibtex

``` -->
