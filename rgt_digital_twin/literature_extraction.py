#
# Copyright [Jul 29, 2025] [Jacqueline Lammert, Maximilian Tschochohei]
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
# This class holds an agent to extract data from scientific literature using
# the Google Gemini 2.5 Flash LLM
#

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# For access to the Gemini model via Vertex AI
from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting, HarmBlockThreshold, HarmCategory, Part

# --- Configuration & Initialization ---

# Load environment variables from a .env file if it exists
load_dotenv()

# Configure logging to capture detailed information during execution
logging.basicConfig(
    filename='literature_extraction.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)

# Get GCP configuration from environment variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("LOCATION")
MODEL_NAME = "gemini-2.5-flash"  # Powerful model suitable for complex extraction


# --- Prompt Templates ---
SYSTEM_INSTRUCTION = """
You are a highly specialized AI assistant for extracting structured information from medical scientific literature.
Your task is to analyze the provided research paper (PDF) and extract specific data points based on the user's request.
You must provide the output strictly in a single, well-formed JSON format.
The keys of the JSON object must be the exact entity names provided in the prompt.
If an entity cannot be found or is not applicable to the paper (e.g., asking for study arms in a case report), use the value "N/A".
Do not include any explanatory text, markdown formatting, or anything else outside of the final JSON object.
"""

PROMPT_TEMPLATE = """
Please extract the following single entity from the attached document.
Use the provided description to guide your extraction.
Respond with ONLY a single JSON object with the entity name as the key.
If the entity cannot be found, the value for the key should be "N/A".

Entities to extract:
- "{entity_key}": "{entity_instruction}"
"""

# --- Core Functions ---

def process_single_pdf(pdf_path: Path, entities_to_extract: dict, client) -> dict:
    """Processes a single PDF file to extract entities using the Gemini model, one entity at a time."""
    print(f"--- Processing PDF: {pdf_path.name} ---")
    log.info(f"Starting processing for {pdf_path.name}")

    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
            pdf_file = Part.from_bytes(data=pdf_data, mime_type="application/pdf")
    except FileNotFoundError:
        log.error(f"File not found: {pdf_path}")
        print(f"Error: File not found at {pdf_path}. Skipping.", file=sys.stderr)
        return {"error": f"File not found: {pdf_path}"}
    except Exception as e:
        log.error(f"Error reading file {pdf_path}: {e}")
        print(f"Error reading file {pdf_path}: {e}. Skipping.", file=sys.stderr)
        return {"error": f"Error reading file: {e}"}

    pdf_results = {}
    safety_settings = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
    ]
    generation_config = GenerateContentConfig(
        temperature=0.2,
        top_p=1.0,
        top_k=32,
        max_output_tokens=2048,  # Reduced for single-entity extraction
        response_mime_type="application/json",  # Enforce JSON output
        system_instruction=SYSTEM_INSTRUCTION,
        safety_settings=safety_settings
    )
    # Loop through each entity and make a separate API call
    for entity_key, entity_instruction in entities_to_extract.items():
        print(f"  - Extracting entity: {entity_key}")

        prompt = PROMPT_TEMPLATE.format(
            entity_key=entity_key,
            entity_instruction=entity_instruction
        )

        try:
            # The API call is now inside the loop
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, pdf_file],
                config=generation_config,
            )

            log.info(f"Successfully received response for entity '{entity_key}' in {pdf_path.name}")

            result_dict = json.loads(response.text)

            # The response should be a simple {"entity_key": "value"}
            extracted_value = result_dict.get(entity_key, "N/A")
            pdf_results[entity_key] = extracted_value
            print(f"    ... Success: Extracted value for {entity_key}")

        except json.JSONDecodeError:
            error_msg = f"Failed to parse JSON response for entity '{entity_key}' in {pdf_path.name}."
            log.error(error_msg)
            if 'response' in locals() and hasattr(response, 'text'):
                log.error(f"Raw response: {response.text}")
                print(f"    ... Warning: {error_msg} Raw response: {textwrap.shorten(response.text, width=100)}", file=sys.stderr)
            else:
                log.error("Raw response not available.")
                print(f"    ... Warning: {error_msg}", file=sys.stderr)
            pdf_results[entity_key] = "ERROR: JSONDecodeError"
        except Exception as e:
            error_msg = f"An error occurred during Gemini API call for entity '{entity_key}' in {pdf_path.name}: {e}"
            log.error(error_msg)
            try:
                if response and response.prompt_feedback:
                    log.error(f"Prompt Feedback: {response.prompt_feedback}")
                    print(f"    ... API Prompt Feedback: {response.prompt_feedback}", file=sys.stderr)
            except (AttributeError, NameError):
                pass
            print(f"    ... Error: {error_msg}", file=sys.stderr)
            pdf_results[entity_key] = "ERROR: API call failed"

    print(f"Finished processing {pdf_path.name}.\n")
    return pdf_results

def save_results_to_json(results: dict, output_file_path: str):
    """Saves the extraction results to a JSON file."""
    print(f"\nSaving results to {output_file_path}...")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully saved extraction results to {output_file_path}")
    except IOError as e:
        log.error(f"Error saving results to {output_file_path}: {e}")
        print(f"\nError saving results to {output_file_path}: {e}", file=sys.stderr)

def main(folder_path: str, entities_file_path: str, output_filename: str):
    """Main function to process all PDF documents in a folder."""
    # Load entities from JSON file
    try:
        with open(entities_file_path, 'r', encoding='utf-8') as f:
            entities_to_extract = json.load(f)
        print(f"Successfully loaded {len(entities_to_extract)} entities from {entities_file_path}.")
    except FileNotFoundError:
        print(f"Error: Entities file not found at {entities_file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {entities_file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if not GCP_PROJECT_ID or not LOCATION:
            raise ValueError("GCP_PROJECT_ID and LOCATION must be set in your .env file or environment.")
        
        # Using the genai.Client is the correct way to interface with Vertex AI
        client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=LOCATION)
        print(f"Successfully initialized Gemini client for model: {MODEL_NAME}")

    except Exception as e:
        log.error(f"Failed to initialize Gemini client: {e}")
        print(f"Error: Failed to initialize Gemini client. Check your GCP configuration and credentials. Details: {e}", file=sys.stderr)
        sys.exit(1)

    pdf_dir = Path(folder_path)
    pdf_files = sorted(list(pdf_dir.glob('*.pdf')))

    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to process in '{folder_path}'.\n")

    all_results = {}
    for pdf_path in pdf_files:
        file_results = process_single_pdf(pdf_path, entities_to_extract, client)
        all_results[pdf_path.name] = file_results

    save_results_to_json(all_results, output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract medical information from all PDF documents in a specified folder using Gemini."
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
        help="The path to the JSON file defining the entities to extract (e.g., literature_entities.json)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="literature_extraction_results.json",
        help="The name of the output JSON file. (default: literature_extraction_results.json)"
    )

    args = parser.parse_args()
    main(folder_path=args.directory, entities_file_path=args.entities, output_filename=args.output)

# Example usage from the command line:
# python literature_extraction.py -d /path/to/your/pdfs -e literature_entities.json -o my_results.json
