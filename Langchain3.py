# -*- coding: utf-8 -*-
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
import os

# Configuration de l'environnement
os.environ["GROQ_API_KEY"] = "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"  # Remplacez par votre clé valide

# Modèle de données avec description en anglais pour éviter les caractères spéciaux
class Movie(BaseModel):
    """Movie representation model"""
    title: str = Field(description="Movie title")
    genre: List[str] = Field(description="Movie genres")
    year: int = Field(description="Release year")

def safe_text(text: str) -> str:
    """Nettoie le texte des caractères problématiques"""
    return text.encode('utf-8', errors='ignore').decode('ascii', errors='ignore')

def initialize_components():
    """Initialise les composants LangChain"""
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7
    )
    
    parser = PydanticOutputParser(pydantic_object=Movie)
    
    prompt_template = PromptTemplate(
        template=safe_text("""
        Recommend a movie based on the query:
        {format_instructions}
        Query: {query}
        """),
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )
    
    return llm, parser, prompt_template

def get_movie_recommendation(query: str) -> Movie:
    """Get structured movie recommendation"""
    llm, parser, prompt_template = initialize_components()
    
    try:
        # Nettoyage de la requête
        safe_query = safe_text(query)
        chain = prompt_template | llm | parser
        return chain.invoke({"query": safe_query})
    except Exception as e:
        print(f"Error processing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Exemple avec requête simple (sans caractères spéciaux)
        recommendation = get_movie_recommendation(
            "A 90s movie with Nicolas Cage"
        )
        
        print("\n=== Movie Recommendation ===")
        print(f"Title: {recommendation.title}")
        print(f"Genres: {', '.join(recommendation.genre)}")
        print(f"Year: {recommendation.year}")
        
    except Exception as e:
        print(f"Failed to get recommendation: {str(e)}")