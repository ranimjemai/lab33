# -*- coding: utf-8 -*-
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
import os

# Configuration
os.environ["GROQ_API_KEY"] = "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"  # Remplacez par votre vraie clé

# Initialisation du modèle Groq
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1024
)

# Outil de recherche
search = DuckDuckGoSearchRun()

# Outil de calcul avec gestion d'erreur améliorée
try:
    math_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Calculate the following precisely. 
        Return ONLY the numerical result without any text.
        Question: {question}"""
    )
    llm_math = LLMMathChain.from_llm(llm=llm, prompt=math_prompt, verbose=True)
except ImportError:
    raise ImportError(
        "LLMMathChain requires numexpr package. Please install with: pip install numexpr"
    )

# Définition des outils
tools = [
    Tool(
        name="Web_Search",
        func=search.run,
        description="Useful for current information lookup"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(llm_math.run(x)),
        description="Useful for precise calculations"
    )
]

# Création de l'agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)

# Exécution
if __name__ == "__main__":
    queries = [
        "Calculate 15% of the population of France",
        "What is 12345 * 67890?",
        "Current population difference between Egypt and Morocco"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = agent.run(query)
            print(f"Result: {result}")
        except Exception as e:
            print(f" Error: {str(e)}")
        print("=" * 60)