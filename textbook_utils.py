from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def build_qa_chain(store):
    # HuggingFace generator
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    llm = HuggingFacePipeline(pipeline=generator)

    # RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",   # simple prompt
        return_source_documents=True
    )
    return qa
