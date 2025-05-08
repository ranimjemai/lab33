# -*- coding: utf-8 -*-
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_groq import ChatGroq

# Configuration de la clé API Groq
os.environ["GROQ_API_KEY"] = "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"

# Initialisation du modèle Groq (avec modèle valide)
llm = ChatGroq(temperature=0.7, model_name="llama3-70b-8192")

# Définition des chaînes
prompt_template1 = PromptTemplate.from_template(
    "List {n} cooking/meal titles for {cuisine} cuisine."
)

# Chaîne 1 - Génération des titres
chain1 = LLMChain(
    llm=llm,
    prompt=prompt_template1,
    output_key="titles"  # Renommé pour correspondre à output_variables
)

# Chaîne 2 - Génération des descriptions
prompt_template2 = PromptTemplate.from_template(
    "Write short descriptions for these meals: {titles}"
)
chain2 = LLMChain(
    llm=llm,
    prompt=prompt_template2,
    output_key="synopsis"  # Renommé pour correspondre à output_variables
)

# Chaîne séquentielle corrigée
complex_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["n", "cuisine"],  # Variables d'entrée corrigées
    output_variables=["titles", "synopsis"],  # Doit matcher les output_key
    verbose=True
)

# Exécution avec les bons paramètres
output = complex_chain({
    "cuisine": "italian",  # Paramètre cohérent avec le template
    "n": 3  # Nombre de plats à générer
})

# Affichage formaté
print("=== Generated Meal Titles ===")
print(output["titles"])
print("\n=== Descriptions ===")
print(output["synopsis"])