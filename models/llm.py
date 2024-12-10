from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

def create_llm_chain():
    GROQ_API_KEY = "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"
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="gemma-7b-it",
        temperature=0.5,
        max_tokens=500
    )
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
        **Question:** {question}
        


        **Guidelines for Answer:**
        - **Accuracy:** Ensure that all information provided is medically accurate and up-to-date.
        - **Clarity:** Explain complex medical terms in simple language that patients can easily understand.
        - **Patient Focus:** Tailor your response to prioritize patient safety and understanding. Provide clear instructions or advice where applicable.
        - **Detailed Explanation:** Offer a detailed explanation of the medicine, including its uses, potential side effects, and any interactions it may have.
        - **Simplified Summary:** Provide a brief, simplified summary of the information for patients who may need a quick overview.
        - **Professional Tone:** Maintain a professional, empathetic, and reassuring tone throughout your response.

        **Final Answer:**
        Please generate a response that combines the detailed medical information, relevant research insights, and patient-friendly advice based on the information provided above. Include information about the Ayurvedic plants in the composition and suggest an alternative medicine or treatment option at the end of your response. Ensure that the response is accurate, clear, and patient-focused. 
        """
    )

    return LLMChain(llm=llm, prompt=prompt_template)

