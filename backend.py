import time, json, requests
from clinic_match import ClinicMatch, ClinicQuery
import pandas as pd

from chatty.LangChain_chatbot_util import *


WATCH_URL = 'https://6d85-155-92-14-110.ngrok-free.app/activeconversations'
OUT_URL = 'https://6d85-155-92-14-110.ngrok-free.app/input?id='
PROVIDER_OUT_URL = 'https://6d85-155-92-14-110.ngrok-free.app/inputProviderDetails?id='


class Backend:
    def __init__(self) -> None:
        LLMFactory.initiate_model(model_name="lmsys/vicuna-7b-v1.3")
        
        key = read_api_key()
        self.cm = ClinicMatch(key)
        self.cq = ClinicQuery(key, LLMFactory.get_model(model_name="lmsys/vicuna-7b-v1.3"))
    
    def start(self):
        print("Starting!")

        while True:
            start = time.time()
            self.loop()
            elapsed = time.time() - start
            
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)
            
    def loop(self):
        site = requests.get(WATCH_URL)
        o = json.loads(site.text)
        for key, conversation in o['conversations'].items():
            if conversation.__len__() % 2 == 1:
                last_response = conversation[-1]
                print("Responding to: " + last_response)
                output = llm_response(last_response)
                
                if output == 'begin provider programatical':
                    print('provider stuff!')
                    Backend.post(self.provider(last_response), PROVIDER_OUT_URL, key)
                else:
                    print('llm stuff!')
                    Backend.post(output, OUT_URL, key)
    
    def provider(self, prompt):
        return self.cm.query(self.cq.query_providers(prompt))

    def post(output, URL, key):
        out_object = { "message": output }
        
        print("Posting {}".format(out_object))
        requests.post(URL + key, json=out_object)