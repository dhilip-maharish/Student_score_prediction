import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%d_%m_%y_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "Logs" , Log_file)
os.makedirs(log_path, exist_ok = True)

Log_file_path = os.path.join(log_path, Log_file)

logging.basicConfig(
    filename = Log_file_path,
    format = "[ %(asctime)s] %(lineno)s - %(name)s -  %(levelname)s - %(message)s",
    level = logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started.")
