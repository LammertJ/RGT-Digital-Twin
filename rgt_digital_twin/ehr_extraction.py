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
# It is optimized for data extraction from .pdf files
#
# Ensure that the following packages are installed on your local machine
#

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
import argparse
import asyncio
import json
from pathlib import Path
import sys

# Initialize the LLM model and prompt template once for efficiency
# Adjust your model name if "gemma3:27b-it-qat" is not the exact identifier
# or if you have a locally customized name.
LLM_MODEL = OllamaLLM(model="gemma3:27b-it-qat") # Or your specific Gemma3 variant name

# Define the prompt template. This version is more explicit about desired output.
EXTRACTION_PROMPT_TEMPLATE = """
Du bist ein hochspezialisierter KI-Assistent für die präzise Extraktion von medizinischen Informationen aus Arztbriefen.
Deine Aufgabe ist es, EXAKT die angeforderte Information aus dem folgenden Textauszug zu extrahieren.
Antworte NUR mit der gesuchten Information im angegebenen Format oder mit "{entity}":"n/a", falls die Information nicht im Text enthalten ist.
Gib KEINE zusätzlichen Erklärungen, Einführungen, den Originaltext oder andere Füllwörter zurück.

--- Textauszug des Dokuments ---
{page_content}
--- Ende des Textauszugs ---

Gesuchte Information: {entity}
Spezifische Anweisungen (inklusive Format): {instructions}

Extraktion:
"""

SELECTION_PROMPT_TEMPLATE = """
Du bist ein hochspezialisierter KI-Assistent für die präzise Extraktion von medizinischen Informationen aus Arztbriefen.
Du erhältst eine Liste von extrahierten Informationen aus einem Arztbrief.
Auf mehreren Seiten wurden Informationen gefunden.
Deine Aufgabe ist es, die RICHTIGE Information aus den Möglichkeiten auszuwählen. Nur eine Information ist korrekt.
Antworte NUR mit der gesuchten Information im angegebenen Format oder mit "{entity}":"n/a", falls die Information nicht im Text enthalten ist.
Gib KEINE zusätzlichen Erklärungen, Einführungen, den Originaltext oder andere Füllwörter zurück.

--- Textauszug des Dokuments ---
{page_content}
--- Ende des Textauszugs ---

Gesuchte Information: {entity}
Spezifische Anweisungen (inklusive Format): {instructions}

Extraktion:
"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
SELECTION_PROMPT = ChatPromptTemplate.from_template(SELECTION_PROMPT_TEMPLATE)


async def load_pdf_page_contents(file_path: str) -> list[str]:
    """
    Asynchronously loads a PDF and returns a list of its page contents as strings.
    """
    loader = PyPDFLoader(file_path)
    page_contents = []
    try:
        async for page_document in loader.alazy_load():
            page_contents.append(page_document.page_content)
    except Exception as e:
        print(f"Error loading or processing PDF {file_path}: {e}")
        raise e
    return page_contents

def extract_entity_from_page_content(
    page_content: str,
    entity: str,
    instructions: str,
    model: OllamaLLM,
    prompt_template: ChatPromptTemplate
) -> str:
    """
    Extracts a specific entity from the given page content using the LLM.
    """
    chain = prompt_template | model | StrOutputParser()

    try:
        response = chain.invoke({
            "entity": entity,
            "instructions": instructions,
            "page_content": page_content
        })
        return response.strip()
    except Exception as e:
        print(f"Error during extraction for entity '{entity}': {e}")
        return f'"{entity}":"n/a"' 
    
def select_entity_from_extract(
    extracted_entities: str,
    entity: str,
    instructions: str,
    model: OllamaLLM,
    prompt_template: ChatPromptTemplate
) -> str:
    """
    Extracts a specific entity from the given page content using the LLM.
    """
    chain = prompt_template | model | StrOutputParser()

    try:
        response = chain.invoke({
            "entity": entity,
            "instructions": instructions,
            "page_content": extracted_entities
        })
        return response.strip()
    except Exception as e:
        print(f"Error during selection for entity '{entity}': {e}")
        return f'"{entity}":"n/a"'

async def process_pdf(pdf_path: Path, entities_to_extract: dict, model: OllamaLLM) -> dict:
    """Processes a single PDF file to extract and select entities."""
    print(f"Processing PDF: {pdf_path.name}\n")
    try:
        document_page_contents = await load_pdf_page_contents(str(pdf_path))
    except Exception:
        return {}

    if not document_page_contents:
        print(f"No content could be extracted from {pdf_path.name}. Skipping.")
        return {}

    print(f"Successfully loaded {len(document_page_contents)} page(s) from {pdf_path.name}.\n")

    all_extracted_data = []  # To store all results from all pages

    for i, current_page_content in enumerate(document_page_contents):
        print(f"--- Processing Page {i + 1} of {pdf_path.name} ---")
        if not current_page_content.strip():
            print("Skipping empty page.")
            continue

        page_results = {}
        for entity_key, entity_instructions in entities_to_extract.items():
            print(f"Attempting to extract: {entity_key}")
            extracted_information = extract_entity_from_page_content(
                page_content=current_page_content,
                entity=entity_key,
                instructions=entity_instructions,
                model=model,
                prompt_template=EXTRACTION_PROMPT
            )
            page_results[entity_key] = extracted_information
        all_extracted_data.append({f"page_{i+1}": page_results})
        print("--- End of Page ---\n")

    final_selected_data = {}
    print(f"\n--- Selecting correct response for {pdf_path.name} ---")
    for entity_key, entity_instructions in entities_to_extract.items():
        print(f" Attempting to select: {entity_key}")
        selected_information = select_entity_from_extract(
            extracted_entities=str(all_extracted_data),
            entity=entity_key,
            instructions=entity_instructions,
            model=model,
            prompt_template=SELECTION_PROMPT
        )
        print(f"Raw model output for {entity_key}: {selected_information}")
        final_selected_data[entity_key] = selected_information

    return final_selected_data

def save_results_to_json(results: dict, output_file_path: str):
    """Saves the extraction results to a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved extraction results to {output_file_path}")
    except IOError as e:
        print(f"\nError saving results to {output_file_path}: {e}", file=sys.stderr)


async def main(folder_path: str, entities_file_path: str, output_filename: str):
    """
    Main function to process all PDF documents in a folder.
    """
    # Load entities from JSON file
    try:
        with open(entities_file_path, 'r', encoding='utf-8') as f:
            entities_to_extract = json.load(f)
    except FileNotFoundError:
        print(f"Error: Entities file not found at {entities_file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {entities_file_path}", file=sys.stderr)
        sys.exit(1)

    # Find all PDF files in the directory
    pdf_dir = Path(folder_path)
    pdf_files = list(pdf_dir.glob('*.pdf'))

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to process in '{folder_path}'.\n")

    all_results = {}
    for pdf_path in pdf_files:
        file_results = await process_pdf(pdf_path, entities_to_extract, LLM_MODEL)
        all_results[pdf_path.name] = file_results

    output_filename = "ehr_extraction_results.json"
    save_results_to_json(all_results, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract medical information from all PDF documents in a specified folder."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="The path to the folder containing PDF documents."
    )
    parser.add_argument(
        "-e", "--entities",
        type=str,
        required=True,
        help="The path to the JSON file defining the entities to extract."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="ehr_extraction_results.json",
        help="The name of the output JSON file. (default: ehr_extraction_results.json)"
    )

    args = parser.parse_args()
    asyncio.run(main(folder_path=args.directory, entities_file_path=args.entities, output_filename=args.output))