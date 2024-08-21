# Introduction
The RGT Digital Twin is an innovative application designed to address the challenges of precision medicine in rare gynecological tumors (RGTs). RGTs are characterized by their low incidence and heterogeneity, leading to difficulties in conducting large-scale clinical trials and establishing standardized treatment guidelines. This often results in suboptimal treatment strategies and poor prognosis for patients.

The application aims to leverage the power of large language models (LLMs) to overcome these challenges. By integrating data from diverse sources, including electronic health records (EHRs) and extensive literature, the RGT Digital Twin can construct personalized digital twins for patients with RGTs. These digital twins model individual patient trajectories, enabling clinicians to identify suitable treatment options, including off-label therapies and clinical trials. The ultimate goal is to advance the management of rare cancers and optimize patient care, particularly in resource-limited settings like Molecular Tumor Boards (MTBs).

# Technical Design
##Data Collection
* The application collects data from institutional EHRs. It accepts any format as it is .pdf, .docx, .jpg or .png. We suggest to put this into a folder `ehr` in the application directory.
* The application also processes literature data, which it accepts in .pdf format. We suggest to put this into a folder `literature` in the application directory.

> [!CAUTION]
> Make sure that you separate EHR and Literature Files! Literature is processed using a cloud-based LLM.

## Data Extraction:
* The application employs a two-stage approach to extract structured, actionable data from source files. It is split in two scripts: `ehr_extraction.py` for extracting patient data from EHR on your local machine, and `literature_extraction.py` to extract relevant information from literature documents.
* A locally deployed LLM system extracts relevant data from institutional EHRs. To ensure patient privacy, it leverages [gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) for local text processing
* A cloud-based LLM system processes documents from web-based repositories. This data is processed using [Gemini 1.5 Pro](https://ai.google.dev/gemini-api/). The script is written in a way that allows you to replace Gemini with any other model
* The extracted data is organized into a structured dataset, which is then made available to clinicians and the locally deployed LLM for further analysis.

## Digital Twin Construction:

* The extracted data points are stored in a database in the secure hospital environment, representing the patient's digital twin. What that means is: We export .csv for you that you can load into any local database.
* Clinicians can adjust filters in the .csv or database, such as biomarker expression or previous treatments, to model potential outcomes and determine suitable treatment strategies.
* The local LLM combines treatment strategies identified in the literature with the patient's digital twin to generate personalized treatment recommendations. Treatment recommendations are extracted by `literature_extraction.py` based on parameters such as disease type and relevant biomarkers (e.g., HER2 status).

> [!CAUTION]
> Do NOT enter patient data or patient personally-identifiable information (PII) into `literature_extraction.py`. It will be processed by a cloud-based LLM!

# How-to
## Prerequisites
### Dependent system packages
To execute RGT Digital Twin, you will need the following packages installed on your local machine to process unstructured text files:
* tesseract-ocr 
* libtesseract-dev 
* tesseract-ocr-deu 
* poppler-utils

### Poetry
Install [Poetry](https://python-poetry.org/docs/) on your local machine for a convenient way to download and manage python packages. 

### Google Cloud resources
You need a Google Cloud project with the Vertex AI API enabled and a `credentials.json` for a service account in your Google Cloud project with Vertex AI user permissions. See the [documentation]((https://cloud.google.com/iam/docs/keys-create-delete)) if you have questions.

And that's it, you're good to go!

## Usage
### Install packages
Run `poetry install` from the package directory.
### Prepare your data
* Move EHR to a folder in your app directory, e.g., `ehr`
* Move relevant scientific literature to a folder in your app directory, e.g., `literature`

> [!CAUTION]
> Make sure that you separate EHR and Literature Files! Literature is processed using a cloud-based LLM.

### EHR Extraction
* First, you execute EHR extraction by extracting text data from EHR and then processing it with your local LLM. We use Py2PDF, Tesseract and python-docx for this, depending on the document type.
> [!CAUTION]
> If you do not have a powerful machine, this will take a *long* time. Make sure you have enough disk space, RAM, and ideally a GPU or two. You can reduce the model size in `config.py`, but this will reduce quality.
* You execute the script with the folder as an argument, e.g., `python rgt-digital-twin/ehr_extraction.py ehr` if your EHR are in folder `ehr` in the package root directory.
* The script will produce a .csv file in the root directory with the extracted information
* All documents that could not be processed (e.g., because the LLM messed up the dictionary format) will be logged in `ehr_extraction.log` so you can add them manually later.

### Literature Extraction
* Next, we process literature data. All .pdf are processed in-context within the LLM, so we do not need to perform any text/image extraction
* You execute the script with the folder and disease information as an argument, e.g., `python rgt-digital-twin/literature_extraction.py literature "disease: uterine carcinosarcoma; biomarker: PD-L1 high, TMB medium, HER2 high"` if your studies are in folder `literature` in the package root directory and you want to discover treatment options for UCS with high PD-L1 and HER2 Status.
> [!CAUTION]
> Do not put confidential patient data or patient personally identifiable information (PII) into the script! The data will be processed by a cloud-based LLM.
* The script will produce a .csv file in the root directory with the extracted information
* All documents that could not be processed (e.g., because the LLM messed up the dictionary format) will be logged in `literature_extraction.log` so you can add them manually later.



