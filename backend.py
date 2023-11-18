import json, requests

from LangChain_chatbot_util import *


WATCH_URL = 'https://6d85-155-92-14-110.ngrok-free.app/activeconversations'
OUT_URL = 'https://6d85-155-92-14-110.ngrok-free.app/input?id='


class Backend:
    def start():
        LLMFactory.initiate_model(model_name="lmsys/vicuna-7b-v1.3")
        
        while True:
           Backend.loop()
            
    def loop():
        site = requests.get(WATCH_URL)
        o = json.loads(site.text)
        for key, conversation in o['conversations'].items():
            if conversation.__len__() % 2 == 1:
                last_response = conversation[-1]
                output = llm_response(last_response)
                
                out_object = { "message": output }
                
                out_json = json.dumps(out_object)
                requests.post(OUT_URL + key, json=out_json)

    def provider():
        ''

Backend.start()