# === File: main.py ===
from analyzer import analyze_emotions, retrieve_context, update_psycho_profile, get_psycho_profile
from responder import generate_response


def main():
    print("🤖 Therapist: Hello, I'm here to listen. What's on your mind?")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye. Take care.")
            break

        # Step 1: Analyze emotions
        emotions = analyze_emotions(user_input)

        # Step 2: Retrieve relevant context via RAG
        try:
            context_chunks = retrieve_context(user_input)
        except ValueError as ve:
            print(f"[Error] {ve}")
            context_chunks = []

        # Step 3: Update psychoanalytic profile based on input and context
        update_psycho_profile(user_input, context_chunks)

        # Step 4: Generate response using profile and user input
        psycho_data = get_psycho_profile()
        response = generate_response(user_input, psycho_data)

        print(f"Therapist: {response}\n")


if __name__ == "__main__":
    main()
