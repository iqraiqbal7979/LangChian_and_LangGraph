import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables (API keys) from .env file
load_dotenv()

def get_summary(information):
    """
    Takes information about a person and returns a summary 
    and two interesting facts using a LangChain chain.
    """
    # Define the prompt template
    summary_template = """
        Given the information about a person: {information}
        
        Please create:
        1. A short summary.
        2. Two interesting facts about them.
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # Initialize the Groq model
    llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")
    
    # Create the chain using LCEL (LangChain Expression Language)
    chain = summary_prompt_template | llm
    
    # Execute the chain
    response = chain.invoke(input={"information": information})
    return response.content

if __name__ == "__main__":
    # Information variable
    information = """
Imran Ahmed Khan Niazi[a] (born 5 October 1952) is a Pakistani former cricketer, philanthropist, and politician who served as the 19th prime minister of Pakistan from August 2018 until April 2022. As a cricketer, he captained the Pakistan national cricket team to victory in the 1992 Cricket World Cup. After retiring from cricket, he founded the Shaukat Khanum Memorial Cancer Hospital and Research Centre, Pakistan's first cancer hospital. He is the founder of the political party Pakistan Tehreek-e-Insaf (PTI) and was its chairman from 1996 to 2023.

Born in Lahore, he graduated from Keble College, Oxford. He began his international cricket career in a 1971 Test series against England. He advocated for neutral umpiring during his captaincy. He led Pakistan to its first-ever Test series victories in India and England during 1987. Playing until 1992, he captained the Pakistan national cricket team for most of the 1980s and early 1990s. In addition to achieving the all-rounder's triple of scoring 3,000 runs and taking 300 wickets in Tests, he holds the world record for the most wickets as a captain in Test cricket, along with the second-best bowling figures in an innings. Moreover, he has won the most Player of the Series awards in Test cricket for Pakistan and ranks fourth overall in Test history. In 2009, he was inducted into the ICC Cricket Hall of Fame.

In his bachelorhood, he had several relationships and was associated with London's nightlife. His first girlfriend, Emma Sergeant, was, according to him, the one woman he truly loved before his first marriage. He had a relationship with Ana-Luisa (Sita) White, daughter of industrialist Gordon White. A California court ruled Khan to be the father of her daughter Tyrian Jade, though he denied paternity. He dated German MTV host Kristiane Backer, introducing her to Islam. He married Jemima Goldsmith in 1995, had two sons, and divorced in 2004 due to her difficulty adjusting to life in Pakistan. He married British-Pakistani journalist Reham Khan in January 2015, but they divorced in October the same year. He married his spiritual guide Bushra Bibi on 18 February 2018.

He supported General Musharraf's 1999 Pakistani coup d'état. His political career involved perceived closeness to the military establishment, including contacts with several ISI chiefs. He became a member of the National Assembly of Pakistan for the first time in the 2002 election. He contested the 2018 Pakistani general election from five constituencies and became the first in Pakistan's electoral history to win all of them. He was elected prime minister in a PTI-led coalition government. During the COVID-19 pandemic, he launched Pakistan's largest welfare programme. In February 2022, he became the first Pakistani prime minister since 2002 to visit Moscow, arriving on the same day the Russian invasion of Ukraine began. During his premiership, he spoke out against Islamophobia in the Western world. In April 2022, he became the first Pakistani prime minister to be removed from office through a no-confidence motion. He alleged US involvement in his removal, blaming Washington for opposing his foreign policy that sought closer relations with China and Russia.

In October 2022, the Election Commission barred him for one term from the National Assembly over the Toshakhana case. In November, he survived an assassination attempt. In May 2023, he was arrested at the Islamabad High Court during a hearing related to the corruption charges; following the May 9 riots, he was released on protective bail a few days later. He was arrested again in August 2023. He has since been sentenced to 14 years in the Al-Qadir Trust case and, as of December 2024, faced 186 cases across Pakistan. He has alleged that his imprisonment is politically motivated, blaming the post-2022 military establishment and the Shehbaz Sharif government, both of which deny the claim.
"""
    
    print("Generating summary...")
    result = get_summary(information)
    print("\n--- AI Response ---\n")
    print(result)