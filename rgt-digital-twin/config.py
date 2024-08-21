###################################################
### Config for EHR Extraction #####################
###################################################
patient_identifier = 'Patient' # We expect Patient records to be in format "patient_identifier-###_"
local_extraction_model = 'google/gemma-27-2b-it'
local_extraction_chunk_size = 5000
local_extraction_overlap = 100

###################################################
### Config for Literature Extraction ##############
###################################################
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

# Define our Google Cloud resources
import os
gcp_project_id = os.environ.get("GCP_PROJECT_ID")
gcp_region = os.environ.get("GCP_REGION")

# Define the generative model
literature_model = GenerativeModel("gemini-1.5-pro-001")

# Set model parameters
generation_config = GenerationConfig(
    temperature=0.2, # 0.2 performs better than in 0.0 in creating valid json
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Set safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}



###################################################
### Prompts #######################################
###################################################

# The prompt for the local model
# Ensure you follow https://ai.google.dev/gemma/docs/formatting 
# Llama notation performs well with Gemma-2
local_extraction_prompt = """<|begin_of_text|>
<|start_header_id|>Task:<|end_header_id|>

Extract key clinical data points from an unstructured patient record and organize them into a Python dictionary.

<|start_header_id|>Instructions:<|end_header_id|>

1. Analyze the patient record provided.
2. Extract the following data points:
    * `gender` (string: "male", "female", or "N/A")
    * `age` (integer or "N/A")
    * `race` (string or "N/A")
    * `diagnosis` (string or "N/A")
    * `number_of_systemic_treatment_lines_no_surgery_or_radiotherapy_without_systemic_treatment` (integer or "N/A")
    * `description_of_previous_systemic_treatment_lines` (string or "N/A")
    * `immune_checkpoint_inhibitor_treatment` (list of strings in the format "[IMMUNE CHECKPOINT INHIBITOR TREATMENT] [OUTCOME] [PROGRESSION FREE SURVIVAL IN MONTHS]", or "N/A")
    * `biomarkers` (list of strings in the format "[MARKER NAME] [VALUE]" or "[MARKER NAME] [HIGH/LOW]", or "N/A")
    * `date_at_first_diagnosis` (date in format "MM/YYYY", or "N/A")
    * `date_of_death` (date in format "MM/YYYY", or "N/A")
    * `overall_survival` (numerical value or "N/A")
3. Store the extracted data in a Python dictionary.
4. Do not include patient names in the output.
5. Only return data fields that you find in the record. If a data point is not mentioned, do not include it in the output dictionary.

<|start_header_id|>Output Format:<|end_header_id|>
A well-formed Python dictionary. 

<|start_header_id|>Example Output:<|end_header_id|>

```python
{
    "age": "44",
    "gender": "female",
    "race": "caucasian",
    "diagnosis": "Iron deficiency",
    "biomarkers": [
        {"PD-L1": "CPS: 41, TPS: 3%, IC: 40%",
        "Tumor Mutational Burden (TMB)": "3.1",
        "MMR": "pMMR (0%)",
        "Other markers": "PIK3CA (p.E545K, 0.26), CHEK2 (p.T367Mfs*15, 0.79) ..."
        },
    ],
    "Previous Systemic Treatments (no surgery and radiotherapy without systemic treatment)": "3 Treatment Lines: Onion Soup (progress after 6 months),
      Tomato Salad (progress after 2 months),
      Celery Steak (progress after 1 month)",
    "ICI (Immune Checkpoint Inhibitor) Treatment": "Candy corn",
    "ICI (Immune Checkpoint Inhibitor) Treatment Response": "PR",
    "Progression-Free Survival with ICI (PFS) (Months)": 7,
}<|end_header_id|>"""

local_summary_prompt = """<|begin_of_text|>
<|start_header_id|>Task:<|end_header_id|>

Extract key data points from a summary of unstructured records and organize them into a Python dictionary.

<|start_header_id|>Instructions:<|end_header_id|>

1. Analyze the record provided.
2. Extract the following data points:
    * `gender` (string: "male", "female", or "N/A")
    * `age` (integer or "N/A")
    * `race` (string or "N/A")
    * `diagnosis` (string or "N/A")
    * `number_of_systemic_treatment_lines_no_surgery_or_radiotherapy_without_systemic_treatment` (integer or "N/A")
    * `description_of_previous_systemic_treatment_lines` (string or "N/A")
    * `immune_checkpoint_inhibitor_treatment` (list of strings in the format "[IMMUNE CHECKPOINT INHIBITOR TREATMENT] [OUTCOME] [PROGRESSION FREE SURVIVAL IN MONTHS]", or "N/A")
    * `biomarkers` (list of strings in the format "[MARKER NAME] [VALUE]" or "[MARKER NAME] [HIGH/LOW]", or "N/A")
    * `date_at_first_diagnosis` (date in format "MM/YYYY", or "N/A")
    * `date_of_death` (date in format "MM/YYYY", or "N/A")
    * `overall_survival` (numerical value or "N/A")
3. Store the extracted data in a Python dictionary.
4. Do not include patient names in the output.
5. Only return data fields that you find in the record. If a data point is not mentioned, do not include it in the output dictionary.

<|start_header_id|>Output Format:<|end_header_id|>
A well-formed Python dictionary. 

<|start_header_id|>Example Output:<|end_header_id|>

```python
{
    "age": "44",
    "gender": "female",
    "race": "caucasian",
    "diagnosis": "Iron deficiency",
    "biomarkers": [
        {"PD-L1": "CPS: 41, TPS: 3%, IC: 40%",
        "Tumor Mutational Burden (TMBsys.argv[1]": "3.1",
        "MMR": "pMMR (0%)",
        "Other markers": "PIK3CA (p.E545K, 0.26), CHEK2 (p.T367Mfs*15, 0.79) ..."
        },
    ],
    "Previous Systemic Treatments (no surgery and radiotherapy without systemic treatment)": "3 Treatment Lines: Onion Soup (progress after 6 months),
      Tomato Salad (progress after 2 months),
      Celery Steak (progress after 1 month)",
    "ICI (Immune Checkpoint Inhibitor) Treatment": "Candy corn",
    "ICI (Immune Checkpoint Inhibitor) Treatment Response": "PR",
    "Progression-Free Survival with ICI (PFS) (Months)": 7,
}<|end_header_id|>"""

literature_extraction_prompt = prompt = """**Task: Generate Structured Study Summaries**

You will assist in summarizing medical scientific studies. Your output should be a well-organized JSON object capturing essential study details.

**Information to Extract:**

* **Study Type**: (case report, clinical study)
* **Patient Population**:
    * Number of participants (n)
    * Race (with percentages if available)
    * Gender (with percentages if available)
    * Age (mean and/or range)
* **Patient Disease**:
    * Diagnosis
    * Stage (if applicable)
    * Metastatic setting (yes/no)
* **Patient Medical History**:
    * Previous treatments
    * Treatment line at study entry
    * Response to prior treatments
* **Biomarker Profile**:
    * Biomarker name and value (high/low or numerical)
* **Treatment & Response**:
    * Treatment administered
    * Treatment response (PR, SD, PD, etc.)
    * Adverse effects (if any)
* **Survival Outcomes**:
    * Median Progression-Free Survival (PFS) (in months)
    * Overall Survival (OS) (in months)

**Specific Instructions**

* **Provide numerical data whenever possible** (percentages, counts, measurements).
* **Return N/A for missing data points.
* **Adapt the structure to the study type.**
    * **Clinical Studies**: Include details for each study arm:
        * Arm name/number
        * Treatment given
        * Patient response
        * PFS
    * **Case Reports**:  Focus on the individual patient's journey through different treatment lines

**Output Format:**

* JSON object with clear labels for each data category.
* For Clinical Studies, use nested JSON objects for each study arm.

**Example Output - Clinical Study**

```json
{
    "Type of study": "Clinical Study",
    "Number of study arms": 3,
    "Study Arms": {
        "Study Arm 1": {
            "n": 38,
            "age": "44-60",
            "gender": "female",
            "race": "caucasian",
            "diagnosis": "Iron deficiency",
            "biomarkers": [
                {"PD-L1": "CPS: 41, TPS: 3%, IC: 40%"},
                {"Tumor Mutational Burden (TMB)": "HIGH"},
                {"EXAMPLE MARKER 3": "LOW"}
            ],
            "Previous Treatments": "Tomato soup",
            "Treatment": "Candy corn",
            "Treatment Response": "PR",
            "Progression-Free Survival (PFS) (Months)": 7,
            "Overall survival (months)": 12
        },
        "Study Arm 2": {
            // ... (data for arm 2)
        },
        "Study Arm 3": {
            // ... (data for arm 3)
        }
    }
}"""


treatment_prompt = """**Task: Generate Precision Oncology Treatment Suggestion**

You will assist doctors treating patients with hard to treat and rare cancers.
Your task is to identify treatment suggestions from literature and clinical trials
based on the patient disease characteristics.
Consider the type of cancer and the biomarkers of the patient.
If biomarkers in the literature lead to successful treatments for patients with similar cancers and are not defined for this patient, suggest that they should be defined.
If you identify recruiting studies that this patient could be eligible for, suggest those as well.
Your output should be a well-organized JSON object capturing essential study details.

**Information to Extract:**

* **Treatment suggestion**: (treatment and medication)
* **Expected treatment response**: (expected response for this patient and reasoning based on the literature including expected progression-free survival and overall survival)
* **Biomarker**: (biomarker that indicates that this treatment could be successful)
* **Biomarker Result**: (biomarker value for this patient, or N/A if the biomarker has not been determined, but should be determined)

**Specific Instructions**

* **Provide numerical data whenever possible** (percentages, counts, measurements).
* **Return N/A for missing data points.

**Output Format:**

* JSON object with clear labels for each data category.

**Patient Data:**
{patient}
"""