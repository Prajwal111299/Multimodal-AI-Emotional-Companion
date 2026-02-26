# Emotion-Aware AI Companion

This project, developed as part of the ECE IDL course at Carnegie Mellon University, is a multimodal empathetic AI therapist and emotional companion. It synthesizes text sentiment and speech emotion to provide emotionally attuned responses to users.

## Project Goal

The primary goal of this project is to create an AI companion that can understand and respond to users' emotional states. It aims to go beyond traditional chatbots by incorporating a deeper level of emotional and psychological analysis, leading to more empathetic and supportive interactions.

## Architecture

The system is built around a centralized LLM agent (powered by OpenAI's GPT-4) and is composed of several key modules:

### 1. User Interface (Streamlit App)

A web-based interface built with Streamlit (`app.py`) allows users to interact with the AI companion through both text and speech.

*   **Speech Input:** The application uses the browser's built-in audio recording capabilities and the OpenAI Whisper API to transcribe the user's speech into text.
*   **Text Input:** Users can also type their messages directly into a text box.
*   **Visualization:** The app provides real-time visualizations of the emotion analysis, including charts and graphs, and displays the conversation history.

### 2. Analyzer Module (`analyzer.py`)

This module is the core of the AI's "mind". It takes the user's input and performs a deep analysis:

*   **Emotion Analysis:** It uses a Hugging Face transformer model (`j-hartmann/emotion-english-distilroberta-base`) to detect a range of emotions from the user's text. The Streamlit app also uses other libraries like `text2emotion`, `nrclex`, and `nltk.sentiment.vader` for a more comprehensive analysis.
*   **Psychoanalysis:** It uses GPT-4 to perform a "psychoanalytic" reading of the user's input, identifying patterns such as cognitive distortions, defense mechanisms, and core beliefs.
*   **User Profile Estimation:** It also uses GPT-4 to make inferences about the user, creating a dynamic profile of their personality and emotional state.

### 3. Responder Module (`responder.py`)

This module generates the AI's response. It takes the user's input, along with the rich context provided by the Analyzer module, and constructs a prompt for the GPT-4 model. The prompt is carefully designed to elicit a warm, empathetic, and reflective response from the AI.

### 4. Speech Emotion Recognition (SER)

The project also includes a custom-built Speech Emotion Recognition (SER) model.

*   **Model:** The SER model is based on a Vision Transformer (ViT) architecture and is trained on mel spectrograms of audio data.
*   **Dataset:** It was trained on the CREMA-D dataset, which contains audio clips of actors expressing various emotions.
*   **Performance:** The model achieved an accuracy of 71.19% on the test set.

*Note: The SER model is not fully integrated into the main application in the current version of the project.*

## Results

The project was evaluated on its ability to generate empathetic responses. The "Emotional Companion AI" was compared to a baseline "Vanilla ChatGPT" model on a set of real-world user inputs from Reddit.

The results show that the **Emotional Companion AI achieved a ~30% higher empathy score** than the baseline model, demonstrating the effectiveness of its architecture.

| Model | BERT Confidence | Semantic Similarity | Empathy Score |
| :--- | :--- | :--- | :--- |
| Vanilla ChatGPT | 0.730 | 0.563 | 0.437 |
| Emotional Companion AI | 0.731 | 0.491 | 0.570 |

The full evaluation results can be found in `emo_comp_eval_results.txt` and `vanilla_eval_results.txt`.

## How to Run the Application

1.  **Install Dependencies:**
    Make sure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install streamlit openai matplotlib nltk transformers text2emotion nrclex python-dotenv sentence-transformers faiss-cpu
    ```
2.  **Set up OpenAI API Key:**
    Create a `.env` file in the project's root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-api-key"
    ```
3.  **Run the Streamlit App:**
    Open a terminal in the project's root directory and run the following command:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

## Future Work

*   **Multimodal Fusion:** Integrate the Speech Emotion Recognition (SER) model into the main application to create a truly multimodal experience.
*   **Real-Time Processing:** Deploy the system in a real-time environment where all modalities are processed simultaneously.
*   **Fine-Tuning Emotional LLMs:** Explore lightweight fine-tuning techniques to further specialize the LLM's behavior based on multimodal emotional input.
*   **Long-Term Memory:** Implement a more sophisticated memory system to allow the AI to remember past conversations and build a more consistent user profile over time.

## Code Availability

The full implementation of this project is available on GitHub: [https://github.com/micahbaldonado/Intro-to-Deep-Learning-Final-Project-AI-Emotional-Companion](https://github.com/micahbaldonado/Intro-to-Deep-Learning-Final-Project-AI-Emotional-Companion)
