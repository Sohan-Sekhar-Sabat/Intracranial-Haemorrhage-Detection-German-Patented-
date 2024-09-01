Intracranial Hemorrhage Detection and Classification Using Deep Learning

**Overview**
  Intracranial Hemorrhage (ICH) is the deadliest form of stroke, with a high mortality rate and no proven medical or surgical treatment. 
  The main subtypes of ICH include intra-parenchymal, intraventricular, subarachnoid, subdural, and epidural. Accurate and rapid diagnosis of ICH is crucial to improving patient outcomes, but the process is complex and time-sensitive, often requiring the expertise of radiologists.
  This project aims to develop an AI-based diagnostic tool to assist in the detection and classification of ICH subtypes, reducing the workload on medical professionals and minimizing human error. The deep learning model, based on the ResNeXt-101 architecture, is fine-tuned using transfer learning techniques. 
  The model is trained on the 2019 Intracranial Hemorrhage dataset provided by the Radiological Society of North America (RSNA).

**Motivation**
  With mortality rates as high as 52% within the first 30 days, early detection of ICH is critical. 
  However, the diagnosis process is complex and time-sensitive, often exacerbated by a shortage of trained professionals. 
  Deep learning offers a solution by automating image interpretation, aiming to improve diagnostic speed and accuracy in emergency settings.

**Problem Statement**
  Diagnosing ICH, particularly in its early stages, is challenging due to the complexity of different hemorrhage subtypes. 
  This project addresses this issue by developing a deep learning model that detects and classifies ICH subtypes from neuroimaging data.

**Model Architecture**
  The model is based on the ResNeXt-101 architecture, fine-tuned with transfer learning. Key features include:
    Pre-trained Weights: Leveraging ImageNet weights for faster training and improved accuracy.
    Custom Classification Layer: Tailored for ICH subtype classification.
    GradCam: Used for visualizing model focus areas during classification.

**Results**
  The model achieved:

  Accuracy: 91.9%
  F1 Score: 77.7%
  Jaccard Index: 72.9%
  Recall: 80.5%
  Precision: 89.1%
These results underscore the model's effectiveness in accurately diagnosing and classifying ICH.

**Dataset**
  The model is trained on the 2019 Intracranial Hemorrhage dataset provided by RSNA, which includes labeled neuroimaging data of various ICH subtypes.
