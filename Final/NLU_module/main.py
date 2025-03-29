from NLU_module.source.model_definition import *
from NLU_module.agents.verifier import Verifier
from NLU_module.agents.adviser import Adviser


class NLU():
    def __init__(self, log_folder='log', file_name='1', with_verifier = True):
        self.path = f'NLU_module/{log_folder}/{file_name}'
        self.history = []
        self.with_verifier = with_verifier
        if self.with_verifier:
            self.verifier = Verifier(model=gpt_35)
        self.adviser = Adviser(model_name='gpt_35') # 'gpt_35', 'starling', 'deepseek'
        self.init = True
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        else:
            with open(f'{self.path}/log.txt', 'w') as file:
                file.write("")
            with open(f'{self.path}/history.txt', 'w') as file:
                file.write("")

    def run(self, contents, context, with_history=True):
        
        user_input = contents
        
        print('________________________________________')
        if self.init:
            response = self.adviser.generate_response(user_input, context, self.history, init=True, with_history=with_history)
            self.init = False
        else:
            response = self.adviser.generate_response(user_input, context, self.history, with_history=with_history)

        with open(f'{self.path}/log.txt', 'a+') as f:
            f.write(f'----------------------- User -----------------------\n{user_input}\n')
            f.write(f'----------------------- Response -----------------------\n{response}\n')

        if self.with_verifier:
            explanation, is_safe = self.verifier.assess_cur_response(response)
            with open(f'{self.path}/log.txt', 'a+') as f:
                f.write('&&&&&&&&&&&&&&&&&&&&&&& Safety &&&&&&&&&&&&&&&&&&&&&&&\n')
                f.write(f'Safety: {is_safe}\n')
                f.write(f'Explanation: {explanation}\n')
            while not is_safe:
                response = self.adviser.generate_response(response, context, self.history, safe=is_safe, explanation=explanation, with_history=with_history)
                explanation, is_safe = self.verifier.assess_cur_response(response)
                with open(f'{self.path}/log.txt', 'a+') as f:
                    f.write(f'----------------------- Response -----------------------\n{response}\n')
                    f.write(f'Safety: {is_safe}\n')
                    f.write(f'Explanation: {explanation}\n')

        self.history.append("User: {}".format(user_input))
        self.history.append("Response: {}".format(response))
        with open(f'{self.path}/history.txt', 'a+') as f:
            f.write(f'------------ User ------------\n{user_input}\n')
            f.write(f'------------ Response ------------\n{response}\n')
        print('Response: \n{}'.format(response))
        print('****************************************')

        return response
