# 🛡️ Internship Shield AI
### Machine Learning-Based Fake Internship & Job Fraud Detection System

Internship Shield AI is a machine learning-powered project designed to identify potentially fraudulent internship and job opportunities by analyzing their textual descriptions.

The project aims to help students and job seekers recognize suspicious opportunities and make more informed decisions before applying.

## 🚀 Project Overview

Finding genuine internships and entry-level jobs can be challenging, especially for students and freshers. Fake recruitment offers, misleading job descriptions, and suspicious internship opportunities can create financial and privacy risks.

**Internship Shield AI** uses Natural Language Processing (NLP) and Machine Learning techniques to analyze job-related text and predict whether an opportunity is potentially fraudulent or legitimate.

## ✨ Key Features

- **Fraud Detection:** Classifies job and internship descriptions using a trained machine learning model.
- **Text Processing:** Uses TF-IDF vectorization to transform textual information into numerical features.
- **ML-Based Prediction:** Uses a trained classification model to generate predictions.
- **Student Safety Focus:** Aims to help students identify potentially suspicious opportunities.
- **Reusable Model:** Includes saved model and vectorizer files for prediction.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Machine Learning | Fraud classification |
| NLP | Text analysis |
| Scikit-learn | Model development and prediction |
| TF-IDF | Text feature extraction |
| Pickle | Saving trained model artifacts |

## 📁 Project Structure

```text
Internship-Shield-AI/
│
├── app.py
├── internship_shield_model.pkl
├── internship_shield_tfidf.pkl
├── project_info.json
└── README.md
```

## ⚙️ How It Works

1. The user provides a job or internship description.
2. The text is processed for machine learning prediction.
3. The TF-IDF vectorizer converts the text into numerical features.
4. The trained classification model analyzes the extracted features.
5. The system generates a fraud/legitimate prediction.

## 💻 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/AmanAgrawalAI/Internship-Shield-AI.git
```

**2. Navigate to the project directory**

```bash
cd Internship-Shield-AI
```

**3. Install the required dependencies**

```bash
pip install pandas numpy scikit-learn streamlit
```

**4. Run the application**

```bash
streamlit run app.py
```

*Note: The commands assume the application uses Streamlit. Additional dependencies may be required depending on the implementation.*

## 📌 Model Artifacts

The repository includes:

- `internship_shield_model.pkl` – Trained machine learning classification model.
- `internship_shield_tfidf.pkl` – Saved TF-IDF vectorizer.
- `project_info.json` – Project configuration and information.
- `app.py` – Application source code.

## 🎯 Project Objectives

- Explore real-world applications of Machine Learning and NLP.
- Identify potentially fraudulent job and internship opportunities.
- Develop a practical AI-based tool focused on student safety.
- Apply text preprocessing, feature extraction, and classification techniques.

## App graph for prediction
<img width="766" height="736" alt="Screenshot 2026-11-27 120258" src="https://github.com/user-attachments/assets/4fb8e30c-c396-45ea-8886-81bb6d0d3dc9" />


## Full prediction display with graph
<img width="1881" height="847" alt="Screenshot 2026-11-27 120326" src="https://github.com/user-attachments/assets/03d4274d-a1af-4baf-aaa9-7f3da30f3418" />


## 🔮 Future Improvements

- Email and recruitment message analysis.
- Detection of suspicious payment requests.
- Company authenticity verification.
- Explainable AI-based risk indicators.
- Improved fraud detection using larger and more diverse datasets.
- Integration with external company verification sources.

## ⚠️ Disclaimer

Internship Shield AI is an experimental machine learning project intended to support awareness and preliminary screening.

Its predictions are not a guarantee that an opportunity is genuine or fraudulent. Users should independently verify employers and recruitment offers before sharing personal information or making payments.
---

## 👨‍💻 Author

**Aman Kumar**
Aspiring Data Scientist | Python · Machine Learning · Computer Vision

[LinkedIn](https://www.linkedin.com/in/aman-kumar-788677378) • [GitHub](https://github.com/AmanAgrawalAI) • [Email](https://mail.google.com/mail/?view=cm&fs=1&to=amanagrawal06922@gmail.com)

## License

Released under the MIT License.

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning & NLP.**
