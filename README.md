
# Evaluation of ChatGPT-4o on Clinical Diagnosis and Decision-Making in Multimodal Cases
## 🎯 Objective
This project evaluated the diagnostic performance, influencing factors, and assistive utility of ChatGPT-4o using a large, real-world dataset of complex multimodal clinical cases from:
- Lancet (Picture Quiz Gallery) (https://www.thelancet.com/picture-quiz)
- NEJM (Image Challenge) (https://www.nejm.org/image-challenge)
- JAMA (Clinical Challenge) (https://jamanetwork.com/collections/44038/clinical-challenge)

Case example:
<p align="center">
  <img src="images/case example.png" width="600">
</p>

## 👣 Evaluation workflow
### 1. Data processing and annotation
Cases were annotated by a team of three clinical experts and six graduate students. 
Extracted metadata included:
- Publication year
- Word count
- Image type
- Image size
- Patient demographics (age and sex)
- Clinical task type (diagnostic vs. non-diagnostic)
- Disease classification (based on ICD-11)
- Specialty
- Anatomical regions
- Difficulty level (for NEJM cases only, categorized as easy, medium, or hard based on historical human respondent accuracy)
### 2. Pilot study
<p align="center">
  <img src="images/pilot study.png" width="600">
</p>

- Dataset: 60 cases randomly selected from the dataset.
- Potential parameters:
  - Manual annotation / Auto annotation / Structured outputs
  - Role-playing prompts / None role-playing prompts
  - Web interface / API
  - API with temperature = 0 / 0.5 / 1
- Metric: Accuracy
### 3. Performance evaluation
**Input format:**
```python
{Images}  
Question: {Question}  
Options: {Option 1}, {Option 2}, {Option 3}, {Option 4}, ...
```

**Evaluation tasks include:**
- Effectiveness evaluation
- Logistic regression analysis of influencing factors
- Image modality contribution evaluation
- AI-assisted decision-making evaluation
