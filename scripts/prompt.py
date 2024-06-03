# PROMPT = """\
# ##System Prompt
# This is a Named Entity Recognition (NER) task for Vietnamese text.
# Given a piece of Vietnamese text, identify and classify the named entities (NE) into three categories: locations, organizations, and persons with these available labels: 'ORGANIZATION_inner', \
# 'PERSON_outer', \
# 'ORGANIZATION_outer', \
# 'LOCATION_outer', \
# 'MISCELLANEOUS_inner', \
# 'LOCATION_inner', \
# 'PERSON_inner', \
# 'MISCELLANEOUS_outer' \
# If no named entities found, return None
# ##Example
# Input sentence: Một sĩ quan cấp cao vừa bị tạm giữ để thẩm vấn về cáo buộc giúp cựu Thủ tướng Thái Lan Yingluck Shinawatra chạy khỏi đất nước.
# Entities with label:
# Thái Lan::LOCATION_outer
# Yingluck::PERSON_outer

# Input sentence: Nỗi đau của người bệnh khi bị tiêm, kéo dài vài giây, vài phút, cùng lắm có thể đến vài ngày.
# Entities with label:
# None

# ##Model Response
# Input sentence: {}
# Entities with label:
# """

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
Each entity should be tagged with its corresponding category (e.g., PERSON_NAME: Nguyen Van Nam, LOCATION: Ha Noi). \
If no named entities found, return None.

**Input sentence**: {}
**Output:**"""