import sys

def error_message_detail(error, error_details:sys):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error  occurred in python script name [{file_name}] linenumber [{exc_tb.tb_lineno}] error message [{str(error)}]"
    
    return error_message 

class CustomException(Exception):
    def __init__(self,error, error_details:sys):
        super().__init__(error)
        self.error = error_message_detail(error, error_details = error_details)
        
    def __str__(self):
        return self.error
            