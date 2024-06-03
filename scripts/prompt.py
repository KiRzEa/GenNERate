PROMPT = """\
##System Prompt
This is a Named Entity Recognition (NER) task for Vietnamese text.
Given a piece of Vietnamese text, identify and classify the named entities (NE) into three categories: locations, organizations, and persons with these available labels: 'ORGANIZATION_inner', \
'PERSON_outer', \
'ORGANIZATION_outer', \
'LOCATION_outer', \
'MISCELLANEOUS_inner', \
'LOCATION_inner', \
'PERSON_inner', \
'MISCELLANEOUS_outer' \
If no named entities found, return None
##Example
Input sentence: Một sĩ quan cấp cao vừa bị tạm giữ để thẩm vấn về cáo buộc giúp cựu Thủ tướng Thái Lan Yingluck Shinawatra chạy khỏi đất nước.
Entities with label:
Thái Lan::LOCATION_outer
Yingluck::PERSON_outer

##Model Response
Input sentence: {}
Entities with label:
"""