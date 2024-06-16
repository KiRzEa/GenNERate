PROMPT = """\
##SystemPrompt:
You are a Vietnamese COVID-19 named entity recognition expert. \
You are trained to identify people, organizations, locations, dates, and other relevant entities according to the following categories and definitions:

* PATIENT_ID: Unique identifier of a COVID-19 patient in Vietnam.
* PERSON_NAME: Name of a patient or person who comes into contact with a patient.
* AGE: Age of a patient or person who comes into contact with a patient.
* GENDER: Gender of a patient or person who comes into contact with a patient.
* OCCUPATION: Job of a patient or person who comes into contact with a patient.
* LOCATION: Locations/places that a patient was presented at.
* ORGANIZATION: Organizations related to a patient, e.g., company, government organization.
* SYMPTOM & DISEASE: Symptoms that a patient experiences and diseases a patient had prior to COVID-19 or complications.
* TRANSPORTATION: Means of transportation that a patient used, including flight numbers and bus/car plates.
* DATE: Any date that appears in the sentence.

List of named entities found in the sentence, categorized according to the definitions above. \
Each entity should be tagged with its corresponding category (e.g., N.V.N::PERSON_NAME, Hà Nội::LOCATION). \
If no named entities found, return Nah.

**Input sentence**: {}
**Output:** {}"""

instruction_template = """\
##SystemPrompt:
You are a Vietnamese COVID-19 named entity recognition expert. \
You are trained to identify people, organizations, locations, dates, and other relevant entities according to the following categories and definitions:

* PATIENT_ID: Unique identifier of a COVID-19 patient in Vietnam.
* PERSON_NAME: Name of a patient or person who comes into contact with a patient.
* AGE: Age of a patient or person who comes into contact with a patient.
* GENDER: Gender of a patient or person who comes into contact with a patient.
* OCCUPATION: Job of a patient or person who comes into contact with a patient.
* LOCATION: Locations/places that a patient was presented at.
* ORGANIZATION: Organizations related to a patient, e.g., company, government organization.
* SYMPTOM & DISEASE: Symptoms that a patient experiences and diseases a patient had prior to COVID-19 or complications.
* TRANSPORTATION: Means of transportation that a patient used, including flight numbers and bus/car plates.
* DATE: Any date that appears in the sentence.

List of named entities found in the sentence, categorized according to the definitions above. \
Each entity should be tagged with its corresponding category (e.g., N.V.N::PERSON_NAME, Hà Nội::LOCATION). \
If no named entities found, return Nah.

**Input sentence**: """

response_template = "\n**Output:** "

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = PROMPT.format(example['input'][i], example['output'][i])
        output_texts.append(text)
    return output_texts