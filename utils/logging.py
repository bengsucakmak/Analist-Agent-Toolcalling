# utils/logging.py
import logging, sys, os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "run.log")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Konsol için format
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Dosya için format (daha detaylı)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    # 1MB'lık 5 log dosyası tutar
    file_handler = RotatingFileHandler(log_file, maxBytes=1*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # Handler'ları ekle (eğer daha önce eklenmemişse)
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
    return logging.getLogger("analist_agent")