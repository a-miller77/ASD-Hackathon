import json, requests
from clinic_match import ClinicMatch, ClinicQuery
import pandas as pd

from LangChain_chatbot_util import *


WATCH_URL = 'https://6d85-155-92-14-110.ngrok-free.app/activeconversations'
OUT_URL = 'https://6d85-155-92-14-110.ngrok-free.app/input?id='


class Backend:
    def start():
        LLMFactory.initiate_model(model_name="lmsys/vicuna-7b-v1.3")
        key = read_api_key()

        cm = ClinicMatch(key)
        cq = ClinicQuery(key)

        while True:
           Backend.loop()
            
    def loop():
        site = requests.get(WATCH_URL)
        o = json.loads(site.text)
        for key, conversation in o['conversations'].items():
            if conversation.__len__() % 2 == 1:
                last_response = conversation[-1]
                output, prompt = llm_response(last_response)
                if output == 'begin provider programatical':
                    Backend.provider(prompt)
                else:
                    Backend.post(output, key)
    
    def provider():
        ''

    def post(output, key):
        out_object = { "message": output }
                
        out_json = json.dumps(out_object)
        requests.post(OUT_URL + key, json=out_json)

Backend.start()