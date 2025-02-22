
import sys
from src.logger import logging


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = f"Error in script: [{file_name}] at line [{exc_tb.tb_lineno}] - [{str(error_message)}]"

    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.error("An error occurred: Divide by zero")
#         raise CustomException(e, sys)

    