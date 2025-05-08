import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# 1. Configurez votre clé API Groq (2 méthodes possibles)

# Méthode 1 : Définir la variable d'environnement (recommandé)
os.environ["GROQ_API_KEY"] = "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"  # Remplacez par votre vraie clé

# Méthode 2 : Ou passez la clé directement au client
# chat_model = ChatGroq(api_key="votre_clé_api_groq_ici", ...)

# 2. Initialisation du modèle Groq
chat_model = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=500,
)

# 3. Définition du message système
system_message = SystemMessage(
    content="You are a friendly pirate who loves to share knowledge. Always respond in pirate speech, use pirate slang, and include plenty of nautical references. Add relevant emojis throughout your responses to make them more engaging. Arr! ☠️🏴‍☠️"
)

# 4. Définition de la question
question = "What are the 7 wonders of the world?"

# 5. Création des messages
messages = [
    system_message,
    HumanMessage(content=question)
]

# 6. Obtention de la réponse
try:
    response = chat_model.invoke(messages)
    
    # 7. Affichage de la réponse
    print("\nQuestion:", question)
    print("\nPirate Response:")
    print(response.content)
    
except Exception as e:
    print(f"Erreur: {str(e)}")
    print("Vérifiez que:")
    print("1. Votre clé API Groq est valide")
    print("2. Le modèle spécifié existe")
    print("3. Vous êtes connecté à internet")