# Depression Analyzer using Deep Learning 🧠

Mental health is a crucial aspect of overall well-being, especially for vulnerable groups such as women and children. With the rise in digital interactions and increased awareness of psychological challenges, there is a growing need for accessible, technology-driven mental health tools.

This project, **Depression Analyzer using Deep Learning**, is built with the vision of early detection and support. It utilizes cutting-edge AI models to analyze multiple data inputs such as:

* **Text responses** from users (e.g., answering a questionnaire or casual chat),
* **Audio inputs**, focusing on tone, pitch, and emotional cues,
* **Facial expressions** captured via image or video analysis.

The deep learning models are trained on validated datasets and psychological scoring systems like PHQ-9, ensuring the analysis aligns with clinical standards. Based on the results, the system generates a depression level score and provides one or more of the following:

* Suggests reaching out to a licensed professional,
* Offers interactive or gamified tasks to uplift mood,
* Connects to emergency mental health helplines.

This tool is designed to be part of a safe, virtual mental wellness environment, where users can interact anonymously and receive non-judgmental support. Though it's not a diagnostic tool, it can serve as a first step toward seeking help or raising awareness about one's mental state.

## 💡 Features
- Detects depression severity using trained deep learning models.
- Accepts inputs via:
  - Text (questionnaire or chat)
  - Voice tone (via audio analysis)
- Outputs depression score or class (normal/mild/moderate/severe).
- Suggests solutions such as:
  - Doctor consultation
  - Gamified activities
  - Helpline contact
- Focused on **mental wellness of women and children**.

## 🧰 Technologies Used
- Python
- TensorFlow / PyTorch
- Natural Language Processing (LSTM / BERT)
- CNN for image/emotion detection
- Audio analysis using MFCC features
- Streamlit / Flask (for web interface)

## 📦 Dataset Sources
- [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)
- [eRisk](https://erisk.irlab.org/)
- PHQ-9 / CES-D questionnaires
- 
## 📚 Research References

This project is supported and inspired by the following research works:

1. **Park et al. (2023)** - *Depression Emotion Multi-Label Classification Using Everytime Platform with DSM-5 Diagnostic Criteria*  
   [IEEE ACCESS, DOI: 10.1109/ACCESS.2023.3305477]  
   > Used Everytime platform data and multi-label classification aligned with DSM-5 standards. Helped guide the emotion labeling strategy.

2. **Skaik & Inkpen (2022)** - *Predicting Depression in Canada by Automatic Filling of Beck’s Depression Inventory Questionnaire*  
   [IEEE ACCESS, DOI: 10.1109/ACCESS.2022.3208470]  
   > Inspired the approach of auto-filling validated questionnaires like BDI using deep learning.

## 🚀 Future Improvements
- Real-time video emotion tracking
- Facial expressions (image/video processing)
- Multi-language support
- Integration with telehealth platforms




