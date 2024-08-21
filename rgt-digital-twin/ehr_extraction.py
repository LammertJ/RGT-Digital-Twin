#
# Copyright [Aug 20, 2024] [Jacqueline Lammert, Maximilian Tschochohei]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This class holds an agent to extract data from electronic health care
# records in a privacy-preserving local environment. 
#
# It is optimized for data extraction from .pdf, .docx, and image files
#
# Ensure that the following packages are installed on your local machine
# tesseract-ocr 
# libtesseract-dev 
# tesseract-ocr-deu 
# poppler-utils
#

import os
import sys
import pytesseract
import docx
import pandas as pd
import ast
import logging
import warnings
import time
import json
import config
#from typing import dict
from pdf2image import convert_from_path
from PyPDF2 import PdfReader


def process_docs(
    filepath: str,
) -> dict:
    """
    Process docs from a filepath and return the output as a dict
    """

    # First check if the directory exists
    try:
        if os.path.exists(filepath):
            docs = os.listdir(os.path.basename(filepath))
    except Exception as e:
        print(f"No valid path given: {e}")

    docs = os.listdir(filepath)
    patients = {}

    # Initialize the patient list; All docs should follow Patient-#### structure
    for doc in docs:
        if config.patient_identifier.lower() in doc.split('_')[0].lower():
            patient = doc.split('_')[0]
            patients[patient] = {}

    print(f"Processing {len(patients)} patients!")

    # Iterate through document links and process the doc
    for doc in docs:

        patient = doc.split('_')[0]
        print(f"Processing {doc}, wish me luck!")
        try:
            # If the doc is a pdf we use PyPDF2 reader (fast and accurate)
            if "pdf" in doc.lower():
                pages = []
                reader = PdfReader(f"{filepath}/{doc}")
                for page in reader.pages:
                    pages.append(page.extract_text())
                patients[patient][doc] = pages

            # If the doc is a docx we use python-docx
            elif "docx" in doc.lower():
                pages = []
                docsux = docx.Document(f"{filepath}/{doc}")
                for para in docsux.paragraphs:
                    pages.append(para.text)
                patients[patient][doc] = pages

            # If the doc is a jpg or image we use pytesseract (slow)
            elif "jpg" in doc.lower() or "png" in doc.lower():
                patients[patient][doc] = [pytesseract.image_to_string(f"{filepath}/{doc}",lang='deu')]

            else:
                raise Exception(f"{doc} did not match document type .pdf, .docx, .jpg, .png")
        except Exception as e:
            print(f"Sorry, I could not process {doc}. Exception: {e}")
    return patients

def extract_attributes(
          patients: dict,
) -> dict:
    """
    Process parsed document input from a dict using a LLM and return a dict with summaries
    """
    
    patient_summaries = {}
    for patient in patients:
        try:
            print(f"Processing {patient}. This may take a while.")
            patient_summaries[patient] = process_document(patients[patient])
        except Exception as e:
            print(f"My apologies, extraction for patient {patient} failed: Exception: {e}")

    return patient_summaries

def process_document(
          document: str,
) -> str:
    """
    Process one document (via a string) and return a string with the summary
    Since most documents are too long, we need to potentially map-reduce
    """
    
    # Review https://github.com/huggingface/local-gemma for gemma-2 local usage instructions
    # Make sure that you load an "instruction-tuned" (it) version of gemma-2
    
    log = logging.getLogger(__name__)

    from local_gemma import LocalGemma2ForCausalLM
    from transformers import AutoTokenizer

    # Initialize the model with memory_extreme for CPU offloading because our local machine is 🐌
    model = LocalGemma2ForCausalLM.from_pretrained(config.local_extraction_model, preset="memory_extreme")
    tokenizer = AutoTokenizer.from_pretrained(config.local_extraction_model)

    # Array for the stuffed document
    stuff = []

    # We initialize i and j for a mini-chunker since the combined patient files are too large for a single model
    i = 0
    j = config.local_extraction_chunk_size

    while i < len(document)-1:
        try:
            prompt = f"""user: "{config.local_extraction_prompt}
                <|start_header_id|>PATIENT RECORDS:<|end_header_id|>\n 
                {document[i:j]}<|end_of_text|> assistant:"""
            model_inputs = tokenizer(prompt, return_attention_mask=True, return_tensors="pt")
            generated_ids = model.generate(**model_inputs.to(model.device), max_length = config.local_extraction_chunk_size)

            response = tokenizer.batch_decode(generated_ids)
            log.info(response)
            stuff.append(response)
        except Exception as e:
            print(f"My apologies, extraction for chunk {j/config.local_extraction_chunk_size} failed: Exception: {e}")

        # If we chunk, leave overlap between chunks so we don't cut words in the middle
        i = j - config.local_extraction_overlap

        # Make sure that our chunk wouldn't be longer than the rest of the doc 
        # So we don't end up out of bounds
        if j + config.local_extraction_chunk_size - config.local_extraction_overlap > len(document):
            j = len(document)-1
        else:
            j = j + config.local_extraction_chunk_size - config.local_extraction_overlap   
    
    # Now we summarize the extracted stuffs
    try:
            prompt = f"""user: "{config.local_summary_prompt}
                <|start_header_id|>PATIENT RECORDS:<|end_header_id|>\n 
                {str(stuff)}<|end_of_text|> assistant:"""
            model_inputs = tokenizer(prompt, return_attention_mask=True, return_tensors="pt")
            generated_ids = model.generate(**model_inputs.to(model.device), max_length = 10000)

            summary = tokenizer.batch_decode(generated_ids)
            log.info(summary)
    except Exception as e:
        print(f"My apologies, summarization failed: Exception: {e}")

    return summary



def export_csv(
    patient_summaries: dict,
    sep: str,
) -> dict:
    """
    Process a dataframe consisting of dictionaries generated by LLMs (this contains errors!)
    To a csv that can be exported to Excel/Sheets for processing by clinicians 
    """ 
    df = pd.DataFrame()
    log = logging.getLogger(__name__)
    # Iterate over patients 
    for patient in patient_summaries:
        success = False
        try:
            # Using ast.literal_eval is inherently unsafe, but we can trust the input here
            results = ast.literal_eval(patient_summaries[patient].split("python")[1].split("```")[0])
            results["Patient"] = patient
            results = pd.DataFrame([results])
            df = pd.concat([df, results], ignore_index=True)
        except Exception as e:
            try:
                # Sometimes the model writes 'json' instead of 'python' 🤷
                results = ast.literal_eval(patient_summaries[patient].split("json")[1].split("```")[0])
                results["Patient"] = patient
                results = pd.DataFrame([results])
                df = pd.concat([df, results], ignore_index=True)
                success = True
            except Exception as f:
                try:
                    # And sometimes we receive nothing at all
                    results = ast.literal_eval(patient_summaries[patient])
                    results["Patient"] = patient
                    results = pd.DataFrame([results])
                    df = pd.concat([df, results], ignore_index=True)
                    success = True
                except Exception as g:
                    print(f"Could not parse record for patient {patient}; Exception: {g}")
                    log.info(patient_summaries[patient])

            if not success:
                print(f"Could not parse record for patient {patient}; Exception: {f}")
                log.info(patient_summaries[patient])
        if not success:
            print(f"Could not parse record for patient {patient}; Exception: {e}")
            log.info(patient_summaries[patient])
    return df.to_csv(sep = sep)

def main():
    import logging
    logging.basicConfig(filename='ehr_extraction.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    
    # First we process the documents using PyPDF2, Tesseract and python-docx
    patients = process_docs(sys.argv[1])
    # Then we extract information from each patient
    attributes = extract_attributes(patients)
    # And finally we parse the extraction into a dataframe and export it as csv
    csv = export_csv(attributes,'|')
    with open("ehr_extracted.csv", "w") as file:
        file.write(csv)

if __name__ == '__main__':
    main() 