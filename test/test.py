from app.rag_pipeline import ask_question

questions = [
    "Who are you?",
     'Qui est le chef de projet de la "Plateforme proactive pour la cyber-sécurité basée sur l\'IA et le Bigdata" ?',
        "Quelle est la date de fin prévue du projet e-commerce sécurisé ?",
        "Combien de chercheurs compose le potentiel humain de la division en 2025 ?",
]

for q in questions:
    result = ask_question(q, session_id=q)
    print("\nQ:", q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])