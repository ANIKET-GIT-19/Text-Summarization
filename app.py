import streamlit as st
import validators
from langchain_classic.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

## Streamlit app
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")


llm=ChatOllama(model="gemma2:latest")
## Get the URL to be summarized
generic_url=st.text_input("URL",label_visibility="collapsed")
prompt="""
Provide a summary of hte following content in 800 words.
content:{text} 
"""
prompt=PromptTemplate(template=prompt,input_variables=["text"])
if st.button("Summarize the content from YT or website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please the provide the URL")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can may be a YT url or web URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Loading the website or YR data:
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False,language=['hi','en'])
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],header_template={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                    
                    ### Chain for summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(docs)
                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")