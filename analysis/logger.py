import datetime
import os

class Logger:
    def __init__(self, log_folder: str) -> None:
        self.log_path = log_folder + f"/logs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
    
    def log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"[[{timestamp}]] {message}"
        print(message)
        with open(self.log_path, "a") as log_file:
            log_file.write(f"{message}\n")